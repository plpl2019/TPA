import sys
from unittest import result
sys.path.append('./yolov4/eval_code') 
sys.path.append('../mmdetection') 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from argparse import ArgumentParser
import torch
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import json
import cv2
import copy
import numpy as np
from tqdm import tqdm
from skimage import measure
import math

from yolov4.eval_code.attack_yolov4 import inference_single_attack_init as yolov4_attack_init
from yolov4.eval_code.attack_yolov4 import inference_single_attack_mt as yolov4_attack_mt
from yolov4.eval_code.tool.darknet2pytorch import *

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().cuda().unsqueeze(0)  


def make_init_mask_img(boxes_init,pred_init,labels_init,mask_layer,img_cv2_800_800, img_cv2,img_path, darknet_model):
    width = img_cv2.shape[3] 
    height = img_cv2.shape[2] 
    attack_map = np.zeros(img_cv2.shape[2:4])
    attack_map_mean = np.zeros(img_cv2.shape[2:4])
    attack_map_mean = np.stack((attack_map_mean, attack_map_mean, attack_map_mean),axis=-1)
    divide_size=[1,2,3]
    size_divide = {0:[],1:[],2:[]}
    for i in range(len(boxes_init)):
        area = np.abs(boxes_init[i][2]-boxes_init[i][0])*np.abs(boxes_init[i][3]-boxes_init[i][1])
        if area<1024:
            size_divide[0].append([boxes_init[i],pred_init[i],labels_init[i]])
        elif area >=1024 and area < 9216:
            size_divide[1].append([boxes_init[i],pred_init[i],labels_init[i]])
        else:
            size_divide[2].append([boxes_init[i],pred_init[i],labels_init[i]])
    for area_size,results in size_divide.items():
        if divide_size[area_size]==1:
            for box in results:
                x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
                w = x2-x1
                h = y2-y1
                attack_map[y1:y2,x1:x2] = 1
                img_mn = region_img_mean(img_cv2,x1,y1,w,h,0,0,1)
                img_mean = img_mn.view(1,1,3)
                attack_map_mean[y1:y2,x1:x2,:] = img_mean
        else:
            if results != []:
                same_class = {}
                for result in results:
                    if same_class.get(result[2]) == None:
                        same_class[result[2]] = [result]
                    else:
                        same_class[result[2]].append(result)
                divide_number = divide_size[area_size]
                for class_name, same_class_box in same_class.items():
                    all_iou = {i:np.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                    all_pred = {i:np.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                    all_score = {i:np.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                    for i in range(divide_number):    
                        for j in range(divide_number):
                            same_class_pred = []
                            img_mask = copy.deepcopy(img_cv2)
                            for box in same_class_box:
                                x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
                                w = x2-x1
                                h = y2-y1
                                same_class_pred.append(box[1])
                                img_mn = region_img_mean(img_cv2,x1,y1,w,h,i,j,divide_number)
                                img_mean = img_mn.view(1,3,1,1)
                                img_mask[:,:,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)] = img_mean
                        
                            boxes_mask , labels_mask,_ = yolov4_attack_init(img_path, darknet_model, img_mask, img_mask)
                            det_pre, det_iou,det_score = mask_img_result_change(same_class_box,same_class_pred,class_name, boxes_mask , labels_mask)
                            for di in range(len(det_iou)):all_iou[di][i,j]=det_iou[di]
                            for dp in range(len(det_pre)):all_pred[dp][i,j]=det_pre[dp]
                            for ds in range(len(det_score)):all_score[ds][i,j]=(1-det_iou[ds])+det_pre[ds]
                            
                    save_pre = pow(divide_number,2)//2
                    for ds in range(len(det_score)):
                        x1,y1,x2,y2 = same_class_box[ds][0][0],same_class_box[ds][0][1],same_class_box[ds][0][2],same_class_box[ds][0][3]
                        w = x2-x1
                        h = y2-y1
                        select_region = k_largest_index_argsort(all_score[ds],save_pre)
                        for sa in range(save_pre):
                            i,j=select_region[sa][0],select_region[sa][1]
                            attack_map[int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)]=1
                            img_mn = region_img_mean(img_cv2,x1,y1,w,h,i,j,divide_number)
                            img_mean = img_mn.view(1,1,3)
                            attack_map_mean[int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number),:]=img_mean
                        
    attack_map = np.stack((attack_map, attack_map, attack_map),axis=-1)       
    return attack_map,attack_map_mean

def mask_img_result_change(same_class_box,same_class_pred,class_name,boxes_mask , labels_mask):
    if class_name not in labels_mask:
        det_pre = np.ones((len(same_class_pred),),dtype=np.float32)
        max_iou = np.zeros((len(same_class_pred),),dtype=np.float32)
        det_score = 1-max_iou+det_pre
    else:  
        same_class_box_mask = boxes_mask[labels_mask==class_name]
        bbox1 = np.array([box[0] for box in same_class_box])
        bbox2 = same_class_box_mask[:,:4]
        boxes_iou = calc_iou(bbox1,bbox2)
        max_iou = np.max(boxes_iou,axis=1)   
        max_index = np.argmax(boxes_iou,axis=1)
        max_pred = same_class_box_mask[:,4][max_index]
        det_pre = same_class_pred-max_pred
        det_score = 1-max_iou+det_pre
    
    return det_pre, max_iou, det_score
    
    
def calc_iou(bbox1,bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)
    
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w
    
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union


def region_img_mean(img_cv2,x1,y1,w,h,i,j,divide_number):
    img_mean = img_cv2.squeeze(0)
    img_mean_0 = torch.mean(img_mean[0,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)])
    img_mean_1 = torch.mean(img_mean[1,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)])
    img_mean_2 = torch.mean(img_mean[2,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)])
    mn = torch.tensor([img_mean_0.item(),img_mean_1.item(),img_mean_2.item()])
    return mn

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

def get_faster_result(img):
    classes = ['airplane','airport','baseballfield','basketballcourt','bridge','chimney','dam',
    'Expressway-Service-area','Expressway-toll-station','golffield',
    'groundtrackfield','harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill']
    # classes=['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    results_path='/disk1/peileipl/programs/mmdetection/results/faster_res50_800/results'
    # results_path='/disk1/peileipl/programs/mmdetection/results/dota/faster_rcnn_r50_fpn_1x_DOTA/results/'
    img_txt=img.split('.')[0]+'.txt'
    results=os.path.join(results_path,img_txt)
    boxes=[]
    labels=[]
    init_det=[]
    with open(results,'r') as f:
        for line in f.readlines():
            score,x1, y1,x2,y2 =line.split(' ')[0].strip(), line.split(' ')[1], line.split(' ')[2], line.split(' ')[3], line.split(' ')[4]
            label_index=classes.index(line.split(' ')[5].strip())
            boxes.append([x1, y1,x2,y2,score])
            labels.append(label_index)
    boxes_out=np.array(boxes,dtype=np.float32)
    labels_out=np.array(labels,dtype=np.int32)
    init_det=torch.tensor(boxes_out,dtype=torch.float32).cuda()
    return boxes_out, labels_out,init_det
    

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_divide', type=int,
                        default=0)
    parser.add_argument('--image_batch', type=str,
                        default='all')
    parser.add_argument('--iters', type=int,
                        default=10)
    parser.add_argument('--save_name', type=str,
                        default='dior_yolo')
    parser.add_argument('--threshold', type=float,
                        default=0)
    args = parser.parse_args()
    return args


def attack_imgs(root_path, imgs):    

    ################## yolov4 init #################
    cfgfile = "yolov4/eval_code/models/yolov4.cfg"
    weightfile = "yolov4/eval_code/models/yolov4-dior_final.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    ################################################

    for ind in range(len(imgs)):
        darknet_model.zero_grad() 
        img_path = os.path.join(root_path, imgs[ind])
        original_img = None
        adversarial_degree = 255.
        ori_bbox_num = None
        attack_map = None 
        img = None
        img_cv2 = toTensor(cv2.imread(img_path))
        img_cv2.requires_grad=True
        img_cv2_800_800 = F.interpolate(img_cv2, (800, 800), mode='bilinear')
        init_det = None
        epsilon = 16/255

        yolo_boxes,labels_pred,_ = yolov4_attack_init(img_path, darknet_model, img_cv2_800_800, img_cv2)
        if yolo_boxes.size==0:
            continue
        with torch.no_grad():
            ########寻找攻击区域########
            if original_img is None:
                original_img = cv2.imread(img_path)
                original_img = np.array(original_img, dtype = np.int16)
                clip_min = np.clip(original_img - adversarial_degree, 0, 255)
                clip_max = np.clip(original_img + adversarial_degree, 0, 255)
            mask_layer=np.zeros(original_img.shape[:2])
            # boxes , labels,init_det = yolov4_attack_init(img_path, darknet_model, img_cv2_800_800, img_cv2)
            boxes , labels,init_det= get_faster_result(imgs[ind])
            ori_bbox_num = len(boxes)
            boxes_init,pred_init,labels_init = [],[],[]
            for i in range(len(labels)):
                if boxes[i][-1]> 0.3:
                    boxes_init.append([int(boxes[i][0]),int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])])
                    pred_init.append(boxes[i][-1])
                    labels_init.append(labels[i])
            attack_map,attack_map_mean=make_init_mask_img(boxes_init,pred_init,labels_init,mask_layer,img_cv2_800_800, img_cv2,img_path, darknet_model)           
            #############################

        
        #########迭代攻击####################
        pbar = tqdm(range(args.iters))
        flag=0
        for attack_iter in pbar:
            if attack_iter != 0:
                if not os.path.exists('./results/{}/iter'.format(args.save_name)):
                    os.makedirs('./results/{}/iter'.format(args.save_name))
                cv2.imwrite(os.path.join('./results/{}/iter'.format(args.save_name), imgs[ind]), img_cv2)
                img_cv2 = toTensor(img_cv2).cuda()
                img_cv2.requires_grad=True
                img_cv2_800_800 = F.interpolate(img_cv2, (800, 800), mode='bilinear')
        
            if original_img is None:
                original_img = cv2.imread(img_path)
                original_img = np.array(original_img, dtype = np.int16)
                clip_min = np.clip(original_img - adversarial_degree, 0, 255)
                clip_max = np.clip(original_img + adversarial_degree, 0, 255)      



            ############### 检测器攻击 ###############            
            iou_thre=args.threshold
            yolo_noise, yolo_boxes,labels_pred,class_loss,iou_loss = yolov4_attack_mt(img_path, darknet_model, img_cv2_800_800, img_cv2,init_det,iou_thre)
            noise_img = np.sign(yolo_noise)
            if np.sum(np.isnan(noise_img))==original_img.size:
                break
            attack_rate =  attack_map[attack_map==1].size / attack_map.size
            output_str = imgs[ind] + '当前{}/{}'.format(imgs.index(imgs[ind]), len(imgs)) + '次数{}'.format(attack_iter)+'最初：{}'.format(ori_bbox_num)+'当前yolo:{}'.format(len(yolo_boxes))+"当前攻击比例:{}".format(attack_rate)
            pbar.set_description(output_str)
            
            img_last = img_cv2.cpu().detach().clone().squeeze(0).numpy().transpose(1, 2, 0)
            img_last = cv2.cvtColor(img_last, cv2.COLOR_RGB2BGR)
            del img_cv2
            a = noise_img.astype(np.float)*attack_map
            a = a[...,::-1].copy()   
            img = np.clip(img_last - a, clip_min, clip_max).astype(np.uint8)
            img_cv2 = copy.deepcopy(img)

        ############## 保存结束 ###############
        save_dir = './results_txt/{}'.format(args.save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_perts = os.path.join(save_dir,'result_{}.txt'.format(args.image_batch))
        img_init = cv2.imread(os.path.join(root_path, imgs[ind])).astype(np.float32)
        img_cv2 = cv2.imread(os.path.join('./results/{}/iter'.format(args.save_name), imgs[ind])).astype(np.float32)
        pp = (img_cv2-img_init)/255
        pp_L2=np.sum((pp) ** 2) ** .5
        pp_Lp=np.max(np.abs(pp))
        pp_L0=pp[pp!=0].size/img_init.size

        with open(save_perts,'a') as f:
            f.write(str(ind)+' '+imgs[ind]+' '+'L2'+' '+str(pp_L2)+' '+'Linf'+' '+str(pp_Lp)+' '+
            'L0'+' '+str(pp_L0)+' '+'iters'+' '+str(attack_iter+1)+'\n')

        

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    root_path = './images/dior/images/'
    imgs = os.listdir(root_path)
    attack_imgs(root_path, imgs)
