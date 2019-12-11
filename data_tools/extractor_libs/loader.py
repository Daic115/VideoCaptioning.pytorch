#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   loader.py
@Desciption     :   None
@Modify Time      @Author    @Version
------------      -------    --------
2019/12/5 15:46   Daic       1.0
'''
import os
import numpy as np
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as trn


def get_universal_transform(setting):
    universal_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(setting['mean'], setting['std'])
    ])
    return universal_transform


def get_scale(sizes):
    scale_transform = trn.Compose([
        trn.Resize(sizes)
    ])
    return scale_transform

def get_sample_ind(sample_mode,all_frame_num):
    num, mode = sample_mode.split('_')
    num = int(num)
    if mode == 'global':
        ind = np.linspace(1, all_frame_num - 4, num, dtype=int)
        return ind
    elif mode == 'stride':
        ind = np.array([i for i in range(1,all_frame_num,num)])
        return ind
    else:
        raise Exception("Unsupported sample mode for '%s'!!"%mode)

class VideoFrames(Data.Dataset):
    def __init__(self, setting, path, sample_mode):
        self.universal_transform = get_universal_transform(setting)
        self.path = path

        self.scale_transform = get_scale((setting['input_size'][1],setting['input_size'][2]))
        self.sample_ind = get_sample_ind(sample_mode,len(os.listdir(path)))


    def __getitem__(self, index):
        id = self.sample_ind[index]
        img = Image.open(os.path.join(self.path, 'image_%05d.jpg' % id)).convert('RGB')

        img = self.scale_transform(img)
        img = self.universal_transform(img)
        return img

    def __len__(self):
        return self.sample_ind.shape[0]