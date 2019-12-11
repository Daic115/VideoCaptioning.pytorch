#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   features_extractor_libs.py
@Desciption     :   This is modified according to https://github.com/hobincar/pytorch-video-feature-extractor/
@Modify Time      @Author    @Version
------------      -------    --------
2019/11/21 10:16   Daic       1.0
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2
import numpy as np
# from abc import ABC

import pretrainedmodels
from efficientnet_pytorch import EfficientNet

'''
This code is modified according to https://github.com/hobincar/pytorch-video-feature-extractor/.
The main changes:
1. using "https://github.com/Cadene/pretrained-models.pytorch" and "https://github.com/lukemelas/EfficientNet-PyTorch" 
   to support more types of backbone CNNs.(pretrainedmodels.__version__==0.7.4)
2. storing data in numpy format
3. feature map extractor is available( use this to implement the attention mechanism based method )
'''

MODEL_SETTING = pretrainedmodels.models.utils.pretrained_settings
for ix in range(8):
    MODEL_SETTING['efficientnet-b' + str(ix)] = {}
    MODEL_SETTING['efficientnet-b' + str(ix)]['imagenet'] = {'input_space':'RGB','num_classes':1000,
                                                             'input_size': [3, 224, 224],
                                                               'std':[0.229, 0.224, 0.225],'mean':[0.485, 0.456, 0.406]}


print("*** Init model setting success! ***")
print("*** Support models:             ***")
for k in MODEL_SETTING.keys():
    print("*** "+k)

class BackboneCnn2D(nn.Module):
    def __init__(self, model_name, pretrained_dataet = 'imagenet', feature_map_flag=False, feature_map_size=7):
        super(BackboneCnn2D, self).__init__()
        self.feature_map_flag = feature_map_flag
        self.model , self.method = BuildFeatureExtractorCnn2D(model_name, pretrained_dataet)
        self.model.eval()

        self.feature_map_resize = nn.AdaptiveAvgPool2d(feature_map_size)
        self.feature_map_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        with torch.no_grad():
            input = getattr(self.model,self.method)(input)

        if self.feature_map_flag:
            output_fc = self.feature_map_pooling(input)
            output_fm = self.feature_map_resize(input)
        else:
            output_fc = self.feature_map_pooling(input)
            output_fm = None

        return output_fc, output_fm

def BuildFeatureExtractorCnn2D(model_name, pretrained_dataset='imagenet'):
    if 'efficientnet' not in model_name:
        return pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained_dataset), 'features'
    else:
        return EfficientNet.from_pretrained(model_name),'extract_features'

