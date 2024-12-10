# 2DPASS Evaluation: Performance Analysis on Waymo and Pandaset

This repository evaluates the **2DPASS** algorithm, originally introduced in [2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds](https://arxiv.org/pdf/2207.04397.pdf) by Yan et al., on additional datasets including **Waymo Open Dataset** and **Pandaset**. This evaluation aims to extend the insights of the algorithm's strengths and limitations in handling various datasets beyond **SemanticKITTI** and **NuScenes**.

![Performance Chart](chart.png)

## Abstract

Our project focuses on enhancing the understanding of semantic segmentation techniques for self-driving cars by evaluating **2DPASS** across diverse datasets. Initially trained on the **SemanticKITTI** dataset, we introduce the **Waymo Open Dataset** and **Pandaset** for further evaluation. Using the same testing methodology from the original 2DPASS implementation, our results provide a comprehensive comparison of its performance, uncovering insights about its generalizability and potential areas of improvement.

## Key Contributions

1. **Dataset Integration**:
   - Converted **Waymo** and **Pandaset** into formats compatible with the 2DPASS algorithm.
   - Integrated the datasets into the training and evaluation pipeline.

2. **Pretrained Evaluation**:
   - Evaluated the performance of pretrained 2DPASS weights on the new datasets.

3. **Model Retraining**:
   - Trained 2DPASS from scratch on **Waymo** and **Pandaset** to benchmark performance differences.

4. **Findings**:
   - Compared results with the original SemanticKITTI and NuScenes benchmarks, highlighting dataset-specific challenges and opportunities for refinement.

## Introduction

2DPASS leverages 2D priors from camera images to assist 3D LiDAR semantic segmentation. Key features include:
- **Multi-Modal Knowledge Distillation**: Integrates 2D and 3D features for improved semantic segmentation.
- **MSFSKD**: Multi-Scale Fusion-to-Single Knowledge Distillation, achieving state-of-the-art results on SemanticKITTI and NuScenes benchmarks.
- **Point Cloud Input**: Uses pure LiDAR point clouds for training, enhancing robustness.

## Methodology

### Evaluation Steps
1. **Dataset Preparation**:
   - Converted **Waymo** and **Pandaset** datasets into the required format for 2DPASS.
   
2. **Testing Pretrained Weights**:
   - Evaluated pretrained weights on the new datasets.

3. **Training on New Datasets**:
   - Trained a new 2DPASS model from scratch on Waymo and Pandaset.

4. **Performance Metrics**:
   - Used **mIoU** and **class accuracy** to evaluate segmentation performance.

### Framework Overview
- **2D and 3D Feature Fusion**:
  - Projects point clouds onto 2D image patches (P2P mapping).
  - Interpolates voxel features onto point clouds (P2V mapping).

- **Training Pipeline**:
  - LiDAR point clouds and cropped image patches generate multi-scale features.
  - These features are fused into a single semantic score using MSFSKD.

![Framework Diagram](figures/2DPASS.gif)

## Experiment Results

### SemanticKITTI Performance
| Metric              | Pretrained Weights on SemanticKITTI |
|----------------------|-------------------------------------|
| **mIoU**            | 72.9%                              |
| **Accuracy**        | 90.1%                              |
| **Road Detection**  | 89.7%                              |
| **Car Detection**   | 97.0%                              |

### Waymo Dataset Performance
| Metric              | Pretrained Weights | Trained on Waymo |
|----------------------|--------------------|------------------|
| **mIoU**            | 8.22%              | 34.9%            |
| **Accuracy**        | 21.2%              | 40.4%            |
| **Road Detection**  | 4.51%              | 0.00%            |
| **Car Detection**   | 35.28%             | 75.38%           |

### Pandaset Performance
| Metric              | Pretrained Weights | Trained on Pandaset |
|----------------------|--------------------|----------------------|
| **mIoU**            | 1.3%               | 57.3%               |
| **Accuracy**        | 7.0%               | N/A                 |
| **Road Detection**  | 14.4%              | 79.9%               |
| **Car Detection**   | 7.6%               | 80.2%               |

### Observations
- **SemanticKITTI** and **NuScenes** benchmarks exhibit excellent performance, especially for common classes like **road**, **car**, and **vegetation**.
- Performance on **Waymo** and **Pandaset** was hindered due to:
  - Differences in LiDAR density and camera resolution.
  - Misaligned camera-LiDAR setups in **Pandaset**.
  - Limited class frequency in training datasets.
- Pretraining on SemanticKITTI generalized poorly to **Waymo** and **Pandaset**, suggesting the need for dataset-specific fine-tuning.

## Discussion and Future Work
1. **Challenges**:
   - Limited transferability across datasets due to inherent differences in sensor setups and data collection conditions.
   - Suboptimal performance on **Pandaset** caused by outdated backbones and misaligned camera-LiDAR calibration.

2. **Proposed Improvements**:
   - **Data Augmentation**: Introduce color augmentation techniques for better generalization.
   - **Model Adaptation**: Leverage techniques like **LoRA** to fine-tune pretrained weights on new datasets.
   - **Backbone Upgrade**: Replace outdated Caffe backbones with modern architectures.
   - **Pose Alignment**: Improve camera-LiDAR pose calibration for **Pandaset**.

## Download Resources
You can download the pretrained weights for evaluation using the links below:
1. **Waymo Converted Dataset**: [Download Link Placeholder]

3. **Pretrained Weights on Waymo**: [Download Link Placeholder]
4. **Pretrained Weights on Pandaset**: [Download](https://umich-my.sharepoint.com/:u:/g/personal/hoangdng_umich_edu/EXCtUtmRf4FInn3FKUiS-GsBjALhGRThsmY4DoszE5DuQQ?e=ueVscc)


## Conversion Code

### Pandaset Conversion
The **Pandaset** conversion code is based on the repository [SiMoM0/Pandaset2Kitti](https://github.com/SiMoM0/Pandaset2Kitti) and has been modified to suit the needs of the 2DPASS evaluation pipeline. The modified conversion script is added to this repository as `/pandaset/convert.py`.

## How to Use

### Running
1. Download and unzip PandasetConv into the `dataset` folder.
2. Place the `pandaset.yaml` configuration file and label map file into their corresponding folders (similar to the SemanticKITTI structure). Ensure the contents and file names are correctly formatted.
3. Put the pretrained checkpoint into the `logs` folder. Pretrained logs should be stored in `logs/SemanticKITTI`.

### Conversion
Conversion is done using the `convert.py` script:
```bash
python convert.py <path_to_pandaset> <path_to_output>
```

## Acknowledgements
Code is forked from [yanx27/2DPASS](https://github.com/yanx27/2DPASS)

Pandaset conversion code is modified from [SiMoM0/Pandaset2Kitti](https://github.com/SiMoM0/Pandaset2Kitti)

PandaSet: https://pandaset.org/
