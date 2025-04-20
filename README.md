<div align="center">

# MARE (Memory-Augmented Re-Completion) 🌕

### [Memory-Augmented Re-Completion for 3D Semantic Scene Completion](https://ojs.aaai.org/index.php/AAAI/article/view/32801)

[Yu-Wen Tseng](https://ywtseng0226.github.io/)<sup>1</sup>,
Sheng-Ping Yang<sup>1</sup>,
[Jhih-Ciang Wu](https://jhih-ciang.github.io/)<sup>1,3</sup>,
I-Bin Liao<sup>4</sup>,
<br>
Yung-Hui Li<sup>4</sup>,
[Hong-Han Shuai](https://basiclab.lab.nycu.edu.tw/)<sup>2</sup>,
[Wen-Huang Cheng](https://www.csie.ntu.edu.tw/~wenhuang/)<sup>4</sup>
<br>

<sup>1</sup> [National Taiwan University](https://www.ntu.edu.tw/),<br>
<sup>2</sup> [National Yang Ming Chiao Tung University](https://www.nycu.edu.tw/nycu/en/index),<br>
<sup>3</sup> [National Taiwan Normal University](https://en.ntnu.edu.tw/),<br>
<sup>4</sup> [Hon Hai Research Institute](https://www.honhai.com/en-us/rd-and-technology/institute),
</div>


[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2311.16090-red)](https://arxiv.org/abs/2409.06355)  -->

## Setup
### Environments
1. Install PyTorch and Torchvision referring to https://pytorch.org/get-started/locally/.
2. Install MMDetection referring to https://mmdetection.readthedocs.io/en/latest/get_started.html#installation.
3. Install the rest of the requirements with pip.

    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

#### 1. Download the Data

**SemanticKITTI:** Download the RGB images, calibration files, and preprocess the labels, referring to the documentation of [VoxFormer](https://github.com/NVlabs/VoxFormer/blob/main/docs/prepare_dataset.md) or [MonoScene](https://github.com/astra-vision/MonoScene#semantickitti).

**SSCBench-KITTI-360:** Refer to https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360.

#### 2. Generate Depth Predications

**SemanticKITTI:** Generate depth predications with pre-trained MobileStereoNet referring to VoxFormer https://github.com/NVlabs/VoxFormer/tree/main/preprocess#3-image-to-depth.

**SSCBench-KITTI-360:** Follow the same procedure as SemanticKITTI but ensure to [adapt the disparity value](https://github.com/ai4ce/SSCBench/issues/8#issuecomment-1674607576).

### Pretrained Weights

Pretrained MaskDINO weights, provided by [Symphonies](https://github.com/hustvl/Symphonies), can be downloaded [here](https://github.com/hustvl/Symphonies/releases/download/v1.0/maskdino_r50_50e_300q_panoptic_pq53.0.pth).