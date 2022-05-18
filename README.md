# LPAE-Net

## Description

This responsitory contains the code of paper "Personalized Outfit Recommendation with Learnable Anchors - CVPR 2021"

## How to

1. clone this responsitory with submodules

   ```bash
   git clone --recurse-submodules https://github.com/lzcn/LPAE-Net.git
   ```

2. install submodules

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

   > This code may looks redundant since I move some implementation to other responsitories. You can simply use and modify the implementation of LPAE-Net to fit your requirement in `models/lpae_net.py` if you like.

3. In each folder of `outfit-datasets`, use the scripts to prepare the dataset. I will improve the `outfit-datasets` so that you can test the model on different datasets that are not used in our paper.

4. Use the `run_lpae_net.py` to train or test.

   - train the LPAE-Net

     ```bash
     # example
     ./run_lpae_net.py train --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml --log-dir summaries/polyvore_630_lpae_u_resnet34_nn --gpus 0
     ```

   - evaluate AUC

     ```bash
     # example
     ./run_lpae_net.py evaluate \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --log-dir summaries/polyvore_630_lpae_u_resnet34_nn \
        --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model.pt \
        --gpus 0 \
        --name evalute-auc
     ```

   - evaluate FITB

     uncomment the following line to evaluate the FITB using corresponding dataset

     ```yaml
     dataset: !include "data-fitb.yaml"
     ```

     ```bash
     # example
     ./run_lpae_net.py evaluate \
        --cfg configs/polyvore_630_lpae_u_resnet34_nn.yaml \
        --log-dir summaries/polyvore_630_lpae_u_resnet34_nn \
        --load-trained summaries/polyvore_630_lpae_u_resnet34_nn/checkpoints/best_model.pt \
        --gpus 0 \
        --name evalute-fitb
     ```

## Contact

Email: zhilu@std.uestc.edu.cn
