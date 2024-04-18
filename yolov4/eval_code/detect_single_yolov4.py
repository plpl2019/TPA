import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from tool.darknet2pytorch import *
from tqdm import tqdm
from skimage import measure

def detection_single_yolov4(img_all,imgs):
    classes = ['airplane','airport','baseballfield','basketballcourt','bridge','chimney','dam',
    'Expressway-Service-area','Expressway-toll-station','golffield',
    'groundtrackfield','harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill']
    cfgfile = "yolov4/eval_code/models/yolov4.cfg"
    weightfile = "yolov4/eval_code/models/yolov4-dior_final.weights"
    
    # classes = ['airplane','airport','baseballfield','basketballcourt','bridge','chimney','dam',
    # 'Expressway-Service-area','Expressway-toll-station','golffield',
    # 'groundtrackfield','harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill']
    # cfgfile = "yolov4/eval_code/models/yolov4.cfg"
    # weightfile = "yolov4/eval_code/models/yolov4-dota_final.weights"
    
    
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    bb_score_dict = {}
   
    for im in imgs:
        img_path=os.path.join(img_all,im)
        img = Image.open(img_path).convert('RGB')
        resize_small = transforms.Compose([
            transforms.Resize((800, 800)),
        ])
        img = resize_small(img)

        boxes = do_detect(darknet_model, img, 0.5, 0.4, True)
        img_txt=img_path.split('/')[-1].split('.')[0]+'.txt'
        save_path='./results/retina_50/txt'
        save_name=os.path.join(save_path,img_txt)
        for box in boxes:
            height=800
            width=800
            x1 = min(max(int((box[0] - box[2] / 2.0) * height), 0), height) # 原图大小不同 需要改
            y1 = min(max(int((box[1] - box[3] / 2.0) * width), 0), width) # 原图大小不同 需要改
            x2 = min(max(int((box[0] + box[2] / 2.0) * height), 0), height) # 原图大小不同 需要改
            y2 = min(max(int((box[1] + box[3] / 2.0) * width), 0), width) # 原图大小不同 需要改
            score=box[4]
            class_name=classes[box[6]]
            with open(save_name,'a') as f:
                f.write(str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+
                str(x1)+' '+str(y2)+' '+str(class_name)+' '+'\n')



if __name__ == '__main__':
    img_all = './results/retina_50/'
    imgs=os.listdir(img_all)
    detection_single_yolov4(img_all,imgs)
