#!/usr/bin/env python
import argparse
import json
import logging
import os
from pprint import pformat

import numpy as np
import outfit_datasets
import torch
import torchutils
import tqdm
from attr import evolve
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler, TensorboardLogger
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from outfit_datasets import RunParam, metrics
from torch import nn
from torch.nn.parallel import data_parallel
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import LPAENet

LOGGER = logging.getLogger("main")


def get_net(param: RunParam, training=False, cuda=False):
    LOGGER.info("Creating net: \n{}".format(param.net_param))
    net = LPAENet(param.net_param)
    if param.load_trained:
        torchutils.load_pretrained(net, param.load_trained)
    if cuda:
        net.cuda()
    if training:
        net.train()
    else:
        net.eval()
    return net


def get_eval_engine(net: nn.Module, dataloader: DataLoader) -> Engine:
    def test_batch(enginer: Engine, batch):
        data = torchutils.to_device(batch["data"])
        uidx = torchutils.to_device(batch["uidx"])
        cate = torchutils.to_device(batch["cate"])
        labels = batch["label"]
        scores = net(data, uidx, cate)
        return scores.tolist(), labels.tolist(), uidx.tolist()

    tester = Engine(test_batch)
    tester.net = net
    tester.state.dataloader = dataloader
    pbar = ProgressBar(desc="Evaluating")
    pbar.attach(tester)
    bundles = dict(
        loss=metrics.functional.pair_rank_loss, ndcg=metrics.functional.ndcg_score, auc=metrics.functional.auc_score
    )
    metrics.EpochBundleMetric(bundles).attach(tester)

    @tester.on(Events.EPOCH_STARTED)
    def setup_tester(enginer: Engine):
        enginer.net.eval()
        enginer.state.dataloader.build()

    return tester


def get_train_engine(config: RunParam, net: nn.Module, tester: Engine):
    tb_writer = SummaryWriter(config.log_dir, flush_secs=10)
    tb_logger = TensorboardLogger(log_dir=config.log_dir)
    optimizer, lr_scheduler = config.optim_param.init_optimizer(net)
    dataloader = outfit_datasets.OutfitLoader(config.train_data_param)
    meter = torchutils.meter.GroupMeter(loss=50, accuracy=50)

    def train_batch(enginer: Engine, batch):
        data = torchutils.to_device(batch["data"])
        uidx = torchutils.to_device(batch["uidx"])
        cate = torchutils.to_device(batch["cate"])
        outputs = net(data, uidx, cate)
        return outputs

    trainer = Engine(train_batch)
    trainer.net = net
    trainer.state.dataloader = dataloader
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler
    # learning rate logger
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.EPOCH_COMPLETED)

    # set timers
    timer = torchutils.ignite.handlers.ModelTimer()
    timer.attach(trainer)

    @trainer.on(Events.EPOCH_STARTED)
    def setup(trainer: Engine):
        trainer.net.train()
        trainer.state.dataloader.build()

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_tester(trainer: Engine):
        tester.run(tester.state.dataloader)
        trainer.state.metrics = tester.state.metrics

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_lr_after_epoch(trainer: Engine):
        if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
            metric = trainer.state.metrics["auc"]
            lr_scheduler.step(metric)
        else:
            lr_scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def summary_metrics(trainer: Engine):
        epoch = trainer.state.epoch
        metrics = trainer.state.metrics
        tb_writer.add_scalar("Loss/Test", metrics["loss"], epoch)
        tb_writer.add_scalar("Accuracy/Test", metrics["accuracy"], epoch)
        rank_metrics = {"auc": metrics["auc"], "ndcg": metrics["ndcg"]}
        tb_writer.add_scalars("Rank/Test", rank_metrics, epoch)
        LOGGER.info("Epoch %d/%d - Test Results:\n%s", epoch, trainer.state.max_epochs, pformat(metrics))
        trainer.state.model_score = metrics["auc"]

    @trainer.on(Events.ITERATION_COMPLETED)
    def backward(trainer: Engine):
        iteration = trainer.state.iteration
        loss, accuracy = trainer.state.output
        loss_dict, loss = torchutils.gather_loss(loss, config.net_param.loss_weight)
        accuracy_dict = torchutils.gather_mean(accuracy)
        net.zero_grad()
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_value_(net.parameters(), 0.5)
        optimizer.step()
        meter.update("loss", loss_dict)
        meter.update("accuracy", accuracy_dict)
        # increase count for nn time
        if iteration % config.display_interval == 0:
            LOGGER.info(
                "Epoch %d/%d - Iteration %d/%d - Time %s",
                trainer.state.epoch,
                trainer.state.max_epochs,
                iteration % len(dataloader),
                len(dataloader),
                timer,
            )
            meter.logging()
        if iteration % config.summary_interval == 0:
            tb_writer.add_scalars("Loss/Train", loss_dict, iteration)
            tb_writer.add_scalars("Accuracy/train", accuracy_dict, iteration)

    checkpoint = ModelCheckpoint(
        os.path.join(config.log_dir, "checkpoints"),
        filename_prefix="best",
        score_function=lambda x: trainer.state.model_score,
        score_name="val_auc",
        global_step_transform=global_step_from_engine(trainer),
        n_saved=5,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {"model": net})
    trainer.checkpoint = checkpoint

    return trainer


def train(config: RunParam):
    """Training tasks."""
    net = get_net(config, training=True, cuda=True)
    LOGGER.info("Validation data: %s", config.valid_data_param)
    valid_data = outfit_datasets.OutfitLoader(config.valid_data_param)
    tester = get_eval_engine(net, valid_data)
    trainer = get_train_engine(config, net, tester)
    trainer.run(trainer.data, max_epochs=config.epochs)
    config.load_trained = trainer.checkpoint.last_checkpoint
    evaluate(config)


@torch.no_grad()
def evaluate(config: RunParam):
    LOGGER.info("Start Compatibility Evaluation")
    net = get_net(config, training=False, cuda=True)
    dataloader = outfit_datasets.OutfitLoader(config.test_data_param)
    tester = get_eval_engine(net, dataloader)
    avg_metrics = dict()
    for _ in range(config.num_runs):
        tester.run(tester.state.dataloader)
        metrics = tester.state.metrics
        LOGGER.critical("Test results:\n%s", pformat(metrics))
        for key, value in metrics.items():
            if key not in avg_metrics:
                avg_metrics[key] = []
            avg_metrics[key].append(value)
    for key, value in avg_metrics.items():
        LOGGER.info("Averaged %s: mean - %.4f, std - %.4f", key, np.mean(value), np.std(value))


@torch.no_grad()
def fitb(config: RunParam):
    def compute_fitb():
        loader.build()
        scores = []
        for batch in tqdm.tqdm(loader, desc="Computing scores"):
            data = torchutils.to_device(batch["data"])
            uidx = torchutils.to_device(batch["uidx"])
            cate = torchutils.to_device(batch["cate"])
            scores += net(data, uidx, cate).flatten().tolist()
        scores = np.array(scores).reshape((-1, data_param.num_fitb_choices))
        acc = (scores.argmax(axis=1) == 0).mean()
        return acc

    LOGGER.info("Start FITB Evaluation")
    # get net
    net = get_net(config, training=False, cuda=True)
    data_param = config.test_data_param
    LOGGER.info("Get data for FITB questions:\n{}".format(data_param))
    loader = outfit_datasets.OutfitLoader(data_param)

    fitb = []
    for n in range(config.num_runs):
        acc = compute_fitb()
        LOGGER.info("FITB Accuracy [%d]/[%d]: %.4f", n + 1, config.num_runs, acc)
        fitb.append(acc)
    LOGGER.info("FITB Accuracy: %.4f +- %.4f", np.mean(fitb), np.std(fitb))


ACTION_FUNS = {
    "train": train,
    "evaluate": evaluate,
    "fitb": fitb,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="LAPE-Net")
    actions = ACTION_FUNS.keys()
    parser.add_argument("action", help="|".join(sorted(actions)))
    parser.add_argument("--name", help="name for logfile", default=None)
    parser.add_argument("--cfg", help="configuration file.")
    parser.add_argument("--log-dir", help="folder to output", default=None)
    parser.add_argument("--load-trained", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num-runs", default=1, type=int)
    args = parser.parse_args()
    action = args.action
    kwargs = torchutils.from_yaml(args.cfg)
    if args.name is None:
        args.name = action
    if args.log_dir:
        kwargs["log_dir"] = args.log_dir
    if args.load_trained:
        kwargs["load_trained"] = args.load_trained
    if args.gpus:
        kwargs["gpus"] = list(map(int, args.gpus.split(",")))
    config = RunParam(**kwargs)
    # update the setting of net from dataset
    config.net_param.num_users = config.data_param.num_users
    config.net_param.num_types = config.data_param.num_types
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # set logger
    logfile = os.path.join(config.log_dir, f"{args.name}.log")
    torchutils.logger.config(stream_level=config.log_level, log_file=logfile, file_mode="w")
    # save configuration
    yamlfile = os.path.join(config.log_dir, f"{args.name}.yaml")
    with open(yamlfile, "w") as f:
        f.write(config.serialize())
    # run
    ACTION_FUNS[action](config)
