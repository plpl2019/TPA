# Threatening Patch Attacks on Object Detection in Optical Remote Sensing Images

## Datasets

The download link of these datasets can be accessed at [AAOD-ORSI](https://github.com/xuxiangsun/AAOD-ORSI).

## Training details of victim detectors

A total of four kinds of detectors are utilized in this paper for the evaluations including Faster R-CNN, RetinaNet, FCOS, and Yolo-v4. For Faster R-CNN, RetinaNet, and FCOS, we use [MMDetection](https://github.com/open-mmlab/mmdetection)(we use mmcv-full=1.4.7=pypi_0, mmdet=2.3.0=dev_0 in this paper) as the main framework. For Yolo-v4, we use the same framework as [RPAttack](https://github.com/VDIGPKU/RPAttack).

The main body of our method is based on the framework of [RPAttack](https://github.com/VDIGPKU/RPAttack). Since I have graduated with a master's degree and started to work, the collation of the code may be delayed for a while. I will do my best to update the code as soon as possible. Below are the training details of the victim detectors leveraged in this paper. Thanks again for browsing this repository and hoping this can help you.


|         model          |    datesets     | epoch | learning rate | batch |     decay     | decay rate | mAP  | recall |
| :--------------------: | :-------------: | :---: | :-----------: | :---: | :-----------: | :--------: | :--: | :----: |
| Faster R-CNN+Resnet50  | DIOR(train+val) |  24   |     0.01      |   4   |    [16,22]    |    0.1     | 88.3 |  90.3  |
| Faster R-CNN+Resnet50  |   DOTA(train)   |  12   |     0.001     |   2   |    [8,10]     |    0.1     | 68.7 |  77.7  |
| Faster R-CNN+Resnet101 | DIOR(train+val) |  24   |     0.01      |   4   |    [16,22]    |    0.1     | 88.6 |  90.9  |
| Faster R-CNN+Resnet101 |   DOTA(train)   |  12   |     0.001     |   2   |    [8,10]     |    0.1     | 68.4 |  76.1  |
|     FCOS+Resnet50      | DIOR(train+val) |  24   |     0.01      |   4   |    [16,22]    |    0.1     | 87.3 |  91.3  |
|     FCOS+Resnet50      |   DOTA(train)   |  12   |     0.001     |   2   |    [8,10]     |    0.1     | 65.7 |  79.1  |
|     FCOS+Resnet101     | DIOR(train+val) |  24   |     0.01      |   4   |    [16,22]    |    0.1     | 87.6 |  91.6  |
|     FCOS+Resnet101     |   DOTA(train)   |  12   |     0.001     |   2   |    [8,10]     |    0.1     | 66.8 |  80.0  |
|   RetinaNet+Resnet50   | DIOR(train+val) |  24   |     0.01      |   4   |    [16,22]    |    0.1     | 87.3 |  92.8  |
|   RetinaNet+Resnet50   |   DOTA(train)   |  12   |     0.001     |   2   |    [8,10]     |    0.1     | 62.2 |  79.5  |
|  RetinaNet+Resnet101   | DIOR(train+val) |  24   |     0.01      |   4   |    [16,22]    |    0.1     | 87.3 |  92.8  |
|  RetinaNet+Resnet101   |   DOTA(train)   |  12   |     0.001     |   2   |    [8,10]     |    0.1     | 64.8 |  81.3  |
|         YOLOv4         | DIOR(train+val) | 30000 |     0.001     |  32   | [20000,25000] |    0.1     | 89.5 |  90.0  |
|         YOLOv4         |   DOTA(train)   | 30000 |     0.001     |  32   | [20000,25000] |    0.1     | 69.7 |  76.8  |

