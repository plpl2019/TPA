import warnings

import copy
import matplotlib.pyplot as plt
import mmcv
import torch
import cv2
import os
import numpy as np
import math
import torchvision
import torch
from PIL import Image  
from torch.autograd import Variable
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes, encode_mask_results, tensor2imgs
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector



def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    # print(img)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img
    

class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        # if isinstance(results['img'], str):
        #     results['filename'] = results['img']
        #     results['ori_filename'] = results['img']
        # else:
        #     results['filename'] = None
        #     results['ori_filename'] = None
        # try:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]) 
            # torchvision.transforms.Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])])
            # torchvision.transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0])])


        img = transform(results['img'].squeeze(0)).unsqueeze(0)
        # except:
        #     img = mmcv.imread(results['img'])
        results['img'] = [img]
        # results['img_fields'] = ['img']
        # try:
            # results['img_shape'] = img.shape
        # except:
        #     print(results['filename'] )
        # results['ori_shape'] = img.shape
        return results

class LoadImage_fcos(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        # if isinstance(results['img'], str):
        #     results['filename'] = results['img']
        #     results['ori_filename'] = results['img']
        # else:
        #     results['filename'] = None
        #     results['ori_filename'] = None
        # try:
        transform = torchvision.transforms.Compose([
            # torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]) 
            # torchvision.transforms.Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])])
            torchvision.transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0])]) 


        img = transform(results['img'].squeeze(0)).unsqueeze(0)
        # except:
        #     img = mmcv.imread(results['img'])
        results['img'] = [img]
        # results['img_fields'] = ['img']
        # try:
            # results['img_shape'] = img.shape
        # except:
        #     print(results['filename'] )
        # results['ori_shape'] = img.shape
        return results

class LoadImage_original(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        try:
            img = results['img_cv2']
        except:
            img = mmcv.imread(results['img'])
        results['img'] = img
        
        results['img_fields'] = ['img']
        try:
            results['img_shape'] = img.shape
        except:
            print(results['filename'] )
        results['ori_shape'] = img.shape
        
        return results

def inference_single_attack_single_init(img, model, img_cv2_800_800, img_cv2):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    model.zero_grad()
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    # test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    if model.cfg['model']['type']=='FCOS':
        test_pipeline = [LoadImage_fcos()]
    else:
        test_pipeline = [LoadImage()]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    results = {}
    results['img_shape'] = np.array([1,3,800,800])
    # in case that there is no padding
    results['pad_shape'] = np.array([1,3,800,800])
    scale_factor = np.array([1, 1, 1, 1],
                                    dtype=np.float32)
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True

    # data = dict(img=img_cv2_800_800,img_metas=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]])
    data = dict(img=img_cv2_800_800,img_metas=[[{'filename': './images/dior_png/images_new/13702.png', 'ori_filename': '13702.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([102.9801, 115.9465, 122.7717], dtype=np.float32), 'std': np.array([1.0, 1.0, 1.0], dtype=np.float32), 'to_rgb': False}}]])

    data = test_pipeline(data)

    model.eval().cuda()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    data['img'][0] = Variable(data['img'][0], requires_grad=True)

    results,  det_bboxes, det_labels,mlvl_scores = model(return_loss=False, rescale=True, **data) # 推理

    init_det = det_bboxes.detach()
    
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    
    return bboxes,labels,init_det

def inference_single_attack_single_box(img, model, img_cv2_800_800, img_cv2):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    model.zero_grad()
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    # test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    if model.cfg['model']['type']=='FCOS':
        test_pipeline = [LoadImage_fcos()]
    else:
        test_pipeline = [LoadImage()]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    results = {}
    results['img_shape'] = np.array([1,3,800,800])
    # in case that there is no padding
    results['pad_shape'] = np.array([1,3,800,800])
    scale_factor = np.array([1, 1, 1, 1],
                                    dtype=np.float32)
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True

    # data = dict(img=img_cv2_800_800,img_metas=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]])
    data = dict(img=img_cv2_800_800,img_metas=[[{'filename': './images/dior_png/images_new/13702.png', 'ori_filename': '13702.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([102.9801, 115.9465, 122.7717], dtype=np.float32), 'std': np.array([1.0, 1.0, 1.0], dtype=np.float32), 'to_rgb': False}}]])
    data = test_pipeline(data)

    model.eval().cuda()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    data['img'][0] = Variable(data['img'][0], requires_grad=True)

    results,  det_bboxes, det_labels,mlvl_scores = model(return_loss=False, rescale=True, **data) # 推理

    # init_det = det_bboxes.detach()
    
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    
    return bboxes,labels

def inference_single_attack_single_mt(img, model, img_cv2_800_800, img_cv2,init_det,iou_thre):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    model.zero_grad()
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if model.cfg['model']['type']=='FCOS':
        test_pipeline = [LoadImage_fcos()]
        img_mm=[[{'filename': './images/dior_png/images_new/13702.png', 'ori_filename': '13702.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([102.9801, 115.9465, 122.7717], dtype=np.float32), 'std': np.array([1.0, 1.0, 1.0], dtype=np.float32), 'to_rgb': False}}]]
    else:
        test_pipeline = [LoadImage()]
        img_mm=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]]

    test_pipeline = Compose(test_pipeline)
    # prepare data
    results = {}
    results['img_shape'] = np.array([1,3,800,800])
    # in case that there is no padding
    results['pad_shape'] = np.array([1,3,800,800])
    scale_factor = np.array([1, 1, 1, 1],
                                    dtype=np.float32)
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True

    # data = dict(img=img_cv2_800_800,img_metas=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]])
    data = dict(img=img_cv2_800_800,img_metas=img_mm)
    data = test_pipeline(data)

    model.eval().cuda()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    data['img'][0] = Variable(data['img'][0], requires_grad=True)

    results,  det_bboxes, det_labels,mlvl_scores = model(return_loss=False, rescale=True, **data) # 推理
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    if len(det_bboxes) == 0:
        loss = disappear_loss(mlvl_scores[:,-1])
        class_loss=loss.item()
        iou_loss=0
    else:
        loss,class_loss,iou_loss = single_loss(det_bboxes,init_det,iou_thre)
    model.zero_grad()
    loss.backward()

    img_metas = data['img_metas'][0]
    noise = data['img'][0].grad.data.cpu().detach().clone().squeeze(0)
    if torch.norm(noise,p=1) != 0:
        noise = (noise / torch.norm(noise,p=1)).numpy().transpose(1, 2, 0)
    else:
        noise = noise.detach().cpu().numpy().transpose(1, 2, 0)
    del loss
    return noise, bboxes, labels,class_loss,iou_loss

def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage_original()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def faster_rcnn_inference(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage_original()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        # result = model(return_loss=False, rescale=True, **data)
        results, cls_score, bbox_pred, det_bboxes, det_labels,= model(return_loss=False, rescale=True, **data) # 推理
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    return bboxes

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    #Get the coordinates of bounding boxes
    xmin1 = box1[:,0].unsqueeze(-1)
    ymin1 = box1[:,1].unsqueeze(-1)
    xmax1 = box1[:,2].unsqueeze(-1)
    ymax1 = box1[:,3].unsqueeze(-1)

    xmin2 = box2[:,0].unsqueeze(-1)
    ymin2 = box2[:,1].unsqueeze(-1)
    xmax2 = box2[:,2].unsqueeze(-1)
    ymax2 = box2[:,3].unsqueeze(-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    ymin = torch.max(ymin1, torch.squeeze(ymin2, dim=-1))
    xmin = torch.max(xmin1, torch.squeeze(xmin2, dim=-1))
    ymax = torch.min(ymax1, torch.squeeze(ymax2, dim=-1))
    xmax = torch.min(xmax1, torch.squeeze(xmax2, dim=-1))
    
    h = torch.max(ymax - ymin, torch.zeros(ymax.shape).cuda())
    w = torch.max(xmax - xmin, torch.zeros(xmax.shape).cuda())
    intersect = h * w
    
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    iou = intersect / union
    return iou

def disappear_loss(cls_score):
    # disappera loss for general detection cls score
    mseloss = torch.nn.MSELoss()
    loss = mseloss(cls_score[cls_score>=0.3], torch.zeros(cls_score[cls_score>=0.3].shape).cuda())  #########
    # loss = 1. - torch.sum(cls_score[cls_score>=0.3]) / torch.numel(cls_score)
    # loss = 1. - torch.nn.BCELoss()(cls_score, torch.zeros(cls_score.shape).cuda())
    return loss

def faster_loss(det_bboxes,init_det,iou_thre):
    cls_score = det_bboxes[:,-1]
    mseloss = torch.nn.MSELoss()
    if cls_score[cls_score>=iou_thre].shape[0]==0:
        class_loss=mseloss(cls_score*0, torch.zeros_like(cls_score).cuda())  #########
        iou_loss = torch.zeros([1]).cuda()
    else:
        class_loss = mseloss(cls_score[cls_score>=iou_thre], torch.zeros(cls_score[cls_score>=iou_thre].shape).cuda())  #########
        box_pred = det_bboxes[:,0:4]
        box_init = init_det[:,0:4]
        pred_iou = bbox_iou(box_pred,box_init)
        iou_loss = torch.sum(pred_iou)/det_bboxes.shape[0]
    loss = class_loss+iou_loss

    return loss,class_loss.item(), iou_loss.item()

def single_loss(det_bboxes,init_det,iou_thre):
    cls_score = det_bboxes[:,-1]
    # det_bboxes=det_bboxes[cls_score>=iou_thre]
    mseloss = torch.nn.MSELoss()
    if cls_score[cls_score>=iou_thre].shape[0]==0:
        class_loss=mseloss(cls_score*0, torch.zeros_like(cls_score).cuda())  #########
        iou_loss = torch.zeros([1]).cuda()
    else:
        class_loss = mseloss(cls_score[cls_score>=iou_thre], torch.zeros(cls_score[cls_score>=iou_thre].shape).cuda())  #########
        box_pred = det_bboxes[:,0:4]
        box_init = init_det[:,0:4]
        pred_iou = bbox_iou(box_pred,box_init)
        iou_loss = torch.sum(pred_iou)/det_bboxes.shape[0]
    loss = class_loss+iou_loss
    # loss = class_loss
    return loss,class_loss.item(),iou_loss.item()

def faster_loss_class(det_bboxes,init_det):
    cls_score = det_bboxes[:,-1]
    mseloss = torch.nn.MSELoss()
    class_loss = mseloss(cls_score[cls_score>=0.3], torch.zeros(cls_score[cls_score>=0.3].shape).cuda())  #########
    # box_pred = det_bboxes[:,0:4]
    # box_init = init_det[:,0:4]
    # pred_iou = bbox_iou(box_pred,box_init)
    # iou_loss = torch.sum(pred_iou)/det_bboxes.shape[0]
    loss = class_loss
    return loss

def adjust_bbox_size(bbox, rate, ori_rate):
    # bbox [[left, top], [right, down]], rate 缩放的比例 rate为2则是缩小两倍
    # return bbox [(left, top), (right, down)] 缩放之后的
    rate += 0.5 # 冗余，使面积之比在0.02以内
    bbox[0][0] *= ori_rate
    bbox[0][1] *= ori_rate
    bbox[1][0] *= ori_rate
    bbox[1][1] *= ori_rate
    middle = (((bbox[1][0] - bbox[0][0]) / 2.0) + bbox[0][0], 
              ((bbox[1][1] - bbox[0][1]) / 2.0) + bbox[0][1])
    k = (bbox[1][1] - bbox[0][1]) / (bbox[1][0] - bbox[0][0])
    # print(middle)
    distance = middle[0] - bbox[0][0]
    # print("原bbox:", bbox)
    if distance > rate:
        distance /= rate
        x_left = (middle[0] - distance) 
        x_right = (middle[0] + distance)
        y_left = (k * (x_left - middle[0]) + middle[1]) 
        y_right = (k * (x_right - middle[0]) + middle[1])
        # print("调整之后的bbox:", (int(x_left), int(y_left)), (int(x_right), int(y_right)))
        # print("面积改变的比例:", pow((x_right - x_left) / (bbox[1][0] - bbox[0][0]), 2))
        return [(int(x_left), int(y_left)), (int(x_right), int(y_right))]
    else:
        return -1 # bbox太小了 放弃该bbox的优化

    

def inference_single_attack_mt(img, model, img_cv2_800_800, img_cv2,init_det,iou_thre):
    """faster rcnn集成攻击接口，输入模型和路径，返回原图大小的grad"""
    model.zero_grad()
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    results = {}
    results['img_shape'] = np.array([1,3,800,800])
    # in case that there is no padding
    results['pad_shape'] = np.array([1,3,800,800])
    scale_factor = np.array([1.6, 1.6, 1.6, 1.6],
                                    dtype=np.float32)
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True

    data = dict(img=img_cv2_800_800, img_metas=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]])
    data = test_pipeline(data)
    model.eval().cuda()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    results, cls_score, bbox_pred, det_bboxes= model(return_loss=False, rescale=True, **data) # 推理
    
    bbox_result, segm_result = results, None
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    if len(det_bboxes) == 0:
        loss = disappear_loss(cls_score[:,-1])
        class_loss=loss.item()
        iou_loss=0
    else:
        loss,class_loss,iou_loss = faster_loss(det_bboxes,init_det,iou_thre)

    loss.backward()
    model.zero_grad()
    img_metas = data['img_metas'][0]
    noise = img_cv2.grad.data.cpu().detach().clone().squeeze(0)
    
    if torch.norm(noise,p=1) != 0:
        noise = (noise / torch.norm(noise,p=1)).numpy().transpose(1, 2, 0)
    else:
        noise = noise.detach().cpu().numpy().transpose(1, 2, 0)
    del loss
    return noise, bboxes, labels,class_loss,iou_loss

def inference_single_attack_init(img, model, img_cv2_800_800, img_cv2):
    """faster rcnn集成攻击接口，输入模型和路径，返回原图大小的grad"""
    model.zero_grad()
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    results = {}
    results['img_shape'] = np.array([1,3,800,800])
    # in case that there is no padding
    results['pad_shape'] = np.array([1,3,800,800])
    scale_factor = np.array([1.6, 1.6, 1.6, 1.6],
                                    dtype=np.float32)
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True

    data = dict(img=img_cv2_800_800, img_metas=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]])
    data = test_pipeline(data)
    model.eval().cuda()
    
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    results, cls_score, bbox_pred, det_bboxes = model(return_loss=False, rescale=True, **data) 

    init_det = det_bboxes.detach()

    bbox_result, segm_result = results, None
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    return bboxes,labels,init_det

def inference_single_attack_box(img, model, img_cv2_800_800, img_cv2):
    """faster rcnn集成攻击接口，输入模型和路径，返回原图大小的grad"""
    model.zero_grad()
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    results = {}
    results['img_shape'] = np.array([1,3,800,800])
    # in case that there is no padding
    results['pad_shape'] = np.array([1,3,800,800])
    scale_factor = np.array([1.6, 1.6, 1.6, 1.6],
                                    dtype=np.float32)
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True

    data = dict(img=img_cv2_800_800, img_metas=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (800, 800, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]])
    data = test_pipeline(data)
    model.eval().cuda()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    results, cls_score, bbox_pred, det_bboxes= model(return_loss=False, rescale=True, **data) # 推理
    bbox_result, segm_result = results, None
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    return bboxes, labels

async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img, bboxes, labels = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.close()
    # plt.show()

def get_bbox_and_label(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img, bboxes, labels = model.show_result(img, result, score_thr=score_thr, show=False)
    return bboxes, labels