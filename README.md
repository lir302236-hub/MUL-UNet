# MUL-UNet: A Lightweight Multi-Weather Image Restoration Network with Enhanced Edge Preservation
> **Abstract:**
Image restoration aims to reconstruct degraded images under adverse weather conditions into high-quality images, providing reliable inputs for computer vision tasks. However, current deep learning image restoration
> algorithms face three major challenges: First, existing approaches primarily focus on single-weather scenarios, struggling to adapt to complex and varying weather conditions; Second, insufficient integration of edge
> priors during image processing leads to loss of edge details in reconstructed images; Third, existing weather removal methods rely on deep network architectures, limiting deployment in resource-constrained scenarios.
> To address these challenges, this study proposes MUL-UNet, a novel multi-task image restoration network with a single-backbone, multi-branch U-shaped architecture. First, a Multi-task Encoder (MTE) with a representation encoder
> head and dual auxiliary branches is proposed to achieve efficient multi-weather image restoration through adaptive weight adjustment for different weather scenarios; Second, a Gradient Feature Fusion Module (GFFM) is proposed to receive multi-scale edge
> feature maps extracted by the representation encoder head in MTE and fuse them with the multi-scale image feature maps extracted by the U-Net encoder in single-backbone through skip connections, enhancing edge detail preservation; Finally, Multi-scale Convolution Block (MConv Block)
> is proposed to replace traditional large kernel convolutions, reducing network parameters while maintaining feature extraction capabilities, thus improving model deployment efficiency. Experimental results show that the proposed model achieves state-of-the-art performance on multiple benchmark
>  datasets.
>


### Network Architecture

<img width="750" height="500" alt="image" src="https://github.com/user-attachments/assets/7dcaf3e5-366d-4e01-a4eb-0a94dbc92413" />



## Getting started

### Install

We test the code on PyTorch 1.12.1 + CUDA 11.3 + cuDNN 8.3.2.

1. Create a new conda environment
```
conda create -n pt1121 python=3.9
conda activate pt1121
```
2. Install dependencies
```
conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Training and Evaluation

### Train

```sh
torchrun --nproc_per_node=4 train.py --model (model name) --train_set (train subset name) --val_set (valid subset name) --exp (exp name) --use_mp --use_ddp
```

### Test

Run the following script to test the trained model:

```sh
python test.py --model (model name)
```
