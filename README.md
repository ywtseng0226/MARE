
# Memory-Augmented Re-Completion for 3D Semantic Scene Completion

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2311.16090-red)](https://arxiv.org/abs/2409.06355)  -->

**Authors:** <br>
[Yu-Wen Tseng](https://ywtseng0226.github.io/),
Sheng-Ping Yang,
[Jhih-Ciang Wu](https://jhih-ciang.github.io/),
I-Bin Liao,
Yung-Hui Li,
[Hong-Han Shuai](https://basiclab.lab.nycu.edu.tw/),
[Wen-Huang Cheng](https://www.csie.ntu.edu.tw/~wenhuang/)

***The 39th AAAI Conference on Artificial Intelligence 2025***

### 🌕 MARE, meaning "sea" in Latin, refers to the lunar maria—the vast, dark plains formed when ancient lava flooded and healed the Moon’s large impact basins. Inspired by this natural metaphor, our method aims to re-complete and expand incomplete 3D scenes, revealing hidden regions and enhancing semantic understanding.

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

## Usage
## Citation

If you find our paper and code useful for your research, please consider giving this repo a star :star: or citing :pencil::

```BibTeX
@inproceedings{tseng2025memory,
  title={Memory-Augmented Re-Completion for 3D Semantic Scene Completion},
  author={Tseng, Yu-Wen and Yang, Sheng-Ping and Wu, Jhih-Ciang and Liao, I-Bin and Li, Yung-Hui and Shuai, Hong-Han and Cheng, Wen-Huang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7446--7454},
  year={2025}
}
```

## Acknowledgements

The development of this project is inspired and informed by [MonoScene](https://github.com/astra-vision/MonoScene), [MaskDINO](https://github.com/IDEA-Research/MaskDINO), [VoxFormer](https://github.com/NVlabs/VoxFormer), and [Symphonies](https://github.com/hustvl/Symphonies). We are thankful to build upon the pioneering work of these projects.

## License

Released under the [MIT](LICENSE) License.
