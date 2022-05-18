# LPAE-Net

## Description

This responsitory contains the code of paper _"Personalized Outfit Recommendation with Learnable Anchors_ - CVPR 2021"

## Train the model

1. Clone this responsitory with submodules

   ```bash
   git clone --recurse-submodules https://github.com/lzcn/LPAE-Net.git
   ```

2. Install submodules

   - `torchutils` is my personal responsitory that contains utilities for PyTorch.

     ```bash
     cd torchutils
     python setup.py install
     ```

   - `outfit-datasets` is another responsitory that contains currently used fashion datasets for outfit recommendation.

     ```bash
     cd outfit-datasets
     python setup.py install
     ```

3. In each folder of `outfit-datasets`, use the scripts to prepare the dataset. I will improve the `outfit-datasets` so that you can test the model on different datasets that are not used in the original paper.

4. Use the `run_lpae_net.py` to train or test.

   - Train LPAE-Net

     ```bash
     ./run_lpae_net.py train \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --log-dir summaries/polyvore_630_lpae_u_resnet34_nn
        --gpus 0 \
        --name train
     ```

   - Evaluate AUC

     ```bash
     ./run_lpae_net.py evaluate \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --log-dir summaries/polyvore_630_lpae_u_resnet34_nn \
        --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model.pt \
        --gpus 0 \
        --name evalute-auc
     ```

   - Evaluate FITB

     uncomment the following line in the configuration file to evaluate the FITB using corresponding dataset

     ```yaml
     dataset: !include "data-fitb.yaml"
     ```

     ```bash
     ./run_lpae_net.py fitb \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --log-dir summaries/polyvore_630_lpae_u_resnet34_nn \
        --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model.pt \
        --gpus 0 \
        --name evalute-fitb
     ```

## Logs

- LPAE-_u_ (_ResNet-34-nn_) Polyvore-_630_

  - [config.yaml](configs/polyvore_630_lpae_u_resnet34_nn.yaml)
  - [train.log](summaries/polyvore_630_lpae_u_resnet34_nn/train.log)

- LPAE-_u_ (_ResNet-34-nn_) Polyvore-_519_

  - [config.yaml](configs/polyvore_519_lpae_u_resnet34_nn.yaml)
  - [train.log](summaries/polyvore_519_lpae_u_resnet34_nn/train.log)

> _ResNet-34-nn_ reprensents the pretrained image features extracted from [ResNet-34](https://arxiv.org/abs/1512.03385), i.e. the backbone is not fine-tuned.

## Contact

email: zhilu@std.uestc.edu.cn
