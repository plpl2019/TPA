import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json
import cv2
import copy

import numpy as np
import os
from tool.darknet2pytorch import *
from tqdm import tqdm
from skimage import measure
from argparse import ArgumentParser

def inference_single_attack_init(img_path, darknet_model, img, img_cv2):
    img_PIL = Image.open(img_path).convert('RGB')
    original_img = copy.deepcopy(img_PIL)
    boxes, labels,init_boxes = do_attack_init(darknet_model, img_path, img, original_img, 0.5, 0.4, img_cv2, True)
    return boxes,labels,init_boxes

def inference_single_attack_mt(img_path, darknet_model, img, img_cv2,init_boxes,iou_thre):
    img_PIL = Image.open(img_path).convert('RGB')
    original_img = copy.deepcopy(img_PIL)

    boxes,labels, noise,class_loss,iou_loss= do_attack(darknet_model, img_path, img, original_img, 0.5, 0.4, img_cv2,init_boxes,iou_thre,True)
    return noise, boxes,labels,class_loss,iou_loss
