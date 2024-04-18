import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tool.darknet2pytorch import *
from tqdm import tqdm
from skimage import measure

def detection_single_yolov4(img_all,imgs):
    classes = ['airplane','airport','baseballfield','basketballcourt','bridge','chimney','dam',
    'Expressway-Service-area','Expressway-toll-station','golffield',
    'groundtrackfield','harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill']
    # classes=['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']  #dota
    cfgfile = "yolov4/eval_code/models/yolov4.cfg"
    weightfile = "yolov4/eval_code/models/yolov4-dior_final.weights"
    # weightfile = "yolov4/eval_code/models/yolov4-dota_final.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (178, 34, 34)
          , (255, 109, 109), (100, 255, 0), (100, 0, 255), (100, 59, 238), (178, 134, 34)
          , (255, 189, 255), (10, 255, 50), (100, 20, 255), (59, 159, 238), (178, 34, 134)
          , (255, 198, 178), (100, 255, 100), (50, 10, 255), (159, 59, 238), (178, 34, 234)
          , (255, 255, 180), (156, 255, 167), (145, 179, 255), (154, 234, 238), (178, 34, 34)
          , (255, 190, 176), (100, 255, 120), (132, 121, 255), (156, 189, 238), (178, 24, 134)
          , (255, 0, 0)]
   
    for im in imgs:
        img_path=os.path.join(img_all,im)
        img = Image.open(img_path).convert('RGB')
        resize_small = transforms.Compose([
            transforms.Resize((800, 800)),
        ])
        img = resize_small(img)

        boxes = do_detect(darknet_model, img, 0.2, 0.4, True)
        img_txt=img_path.split('/')[-1].split('.')[0]+'.txt'
        save_path='./results/show/dior'
        save_name=os.path.join(save_path,img_txt)
        imag_view=cv2.imread(img_path)
        for box in boxes:
            height=800
            width=800
            x11 = min(max(int((box[0] - box[2] / 2.0) * height), 0), height) # 原图大小不同 需要改
            y11 = min(max(int((box[1] - box[3] / 2.0) * width), 0), width) # 原图大小不同 需要改
            x22 = min(max(int((box[0] + box[2] / 2.0) * height), 0), height) # 原图大小不同 需要改
            y22 = min(max(int((box[1] + box[3] / 2.0) * width), 0), width) # 原图大小不同 需要改
            score=box[4]
            label=classes[box[6]]
            color = colors[classes.index(label)]
            x1=min(x11,x22)
            x2=max(x11,x22)
            y1=min(y11,y22)
            y2=max(y11,y22)
            px1, py1, px2, py2, px3, py3, px4, py4 = x1, y1,x2,y1,x2,y2,x1,y2
            imag_view = cv2.line(imag_view, (int(float(px1)), int(float(py1))), (int(float(px2)), int(float(py2))), color, 3, 3)
            imag_view = cv2.line(imag_view, (int(float(px2)), int(float(py2))), (int(float(px3)), int(float(py3))), color, 3, 3)
            imag_view = cv2.line(imag_view, (int(float(px3)), int(float(py3))), (int(float(px4)), int(float(py4))), color, 3, 3)
            imag_view = cv2.line(imag_view, (int(float(px4)), int(float(py4))), (int(float(px1)), int(float(py1))), color, 3, 3)
            # imag_view = cv2.putText(imag_view, label, (int(float(px4)+15), int(float(py4)+15)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        save_name='my'+'_'+im.split('.')[0]+'_'+'yolov4'+'.png'
        cv2.imwrite(os.path.join('./results/show/dior', save_name), imag_view)


if __name__ == '__main__':
    img_all = './results/images/'
    imgs = ['12886.png']
    detection_single_yolov4(img_all,imgs)
