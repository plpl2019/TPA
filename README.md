# Threatening Patch Attacks on Object Detection in Optical Remote Sensing Images

## Datasets
To access the datasets leveraged in this paper (i.e., DIOR-A and DOTA-A), please refer to [AAOD-ORSI](https://github.com/xuxiangsun/AAOD-ORSI)

## Victim Detectors
A total of four kinds of detectors are utilized in this paper for the evaluations including Faster R-CNN, RetinaNet, FCOS, and Yolo-v4. For Faster R-CNN, RetinaNet, and FCOS, we use [MMDetection](https://github.com/open-mmlab/mmdetection) as the main framework. For Yolo-v4, we use the same framework as [RPAttack](https://github.com/VDIGPKU/RPAttack), which is based on [DarkNet](https://github.com/AlexeyAB/darknet).

## Training Settings and Results on Clean Images
We report the training settings and the baselines of those victim detectors in Tab. 1. Here, the parameters of Decay [a, b] denotes decaying the current LR with the scale factor of Ratio at the a-th and b-th epochs/iters.

<div class="center">

| Model | Datesets | Epoch/Iter | Init LR | Batch | Decay | Rate | mAP | Recall |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Faster R-CNN+Resnet50 | DIOR(train+val) | 24 | 0.01 | 4 | [16,22] | 0.1 | 88.3 | 90.3|
| Faster R-CNN+Resnet50 | DOTA(train) | 12 | 0.001 | 2 | [8,10] | 0.1 | 68.7 | 77.7|
| Faster R-CNN+Resnet101 | DIOR(train+val) | 24 | 0.01 | 4 | [16,22] | 0.1 | 88.6 | 90.9|
| Faster R-CNN+Resnet101 | DOTA(train) | 12 | 0.001 | 2 | [8,10] | 0.1 | 68.4 | 76.1|
| FCOS+Resnet50 | DIOR(train+val) | 24 | 0.01 | 4 | [16,22] | 0.1 | 87.3 | 91.3|
| FCOS+Resnet50 | DOTA(train) | 12 | 0.001 | 2 | [8,10] | 0.1 | 65.7 | 79.1|
| FCOS+Resnet101 | DIOR(train+val) | 24 | 0.01 | 4 | [16,22] | 0.1 | 87.6 | 91.6|
| FCOS+Resnet101 | DOTA(train) | 12 | 0.001 | 2 | [8,10] | 0.1 | 66.8 | 80.0|
| RetinaNet+Resnet50 | DIOR(train+val) | 24 | 0.01 | 4 | [16,22] | 0.1 | 87.3 | 92.8|
| RetinaNet+Resnet50 | DOTA(train) | 12 | 0.001 | 2 | [8,10] | 0.1 | 62.2 | 79.5|
| RetinaNet+Resnet101 | DIOR(train+val) | 24 | 0.01 | 4 | [16,22] | 0.1 | 87.3 | 92.8|
| RetinaNet+Resnet101 | DOTA(train) | 12 | 0.001 | 2 | [8,10] | 0.1 | 64.8 | 81.3|
| YOLOv4 | DIOR(train+val) | 30000 (iters) | 0.001 | 32 | [20000,25000] | 0.1 | 89.5 | 90.0|
| YOLOv4 | DOTA(train) | 30000 (iters) | 0.001 | 32 | [20000,25000] | 0.1 | 69.7 | 76.8|

</div> 


## Methods

### The main codes will be released once accpeted...

## Citation
```
@article{sun2023threatening,
  title={Threatening Patch Attacks on Object Detection in Optical Remote Sensing Images},
  author={Sun, Xuxiang and Cheng, Gong and Pei, Lei and Li, Hongda and Han, Junwei},
  journal={arXiv preprint arXiv:2302.06060},
  year={2023}
}
```
