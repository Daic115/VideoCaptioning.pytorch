#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   main.py
@Desciption     :   None
@Modify Time      @Author    @Version
------------      -------    --------
2019/12/4 18:14   Daic       1.0
'''
import os
import torch
import argparse
import numpy as np
from extractor_libs import *
from tqdm import tqdm
import torch.utils.data as Data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,   required=True,         help="The directory path of videos.")
    parser.add_argument('--batch_size', type=int,   default=8,             help="The directory path of videos.")
    parser.add_argument('--model_type', type=str,   default='2d',          help="2d or 3d models")
    parser.add_argument('--model',      type=str,   default='resnext101',  help="The name of model from which you extract features.")
    parser.add_argument('--sample_mode',type=str,   default='5_stride',    help="sample by 'n_global' or 'n_stride' ")
    parser.add_argument('--out_path',   type=str,   required=True,         help="The output path of extracted feature.")
    return parser.parse_args()

def extract_2d_features(args):
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    extractor = BackboneCnn2D(args.model)
    print("** using model        : %s" % args.model)
    print("** sample mode        : %s" % args.sample_mode)
    print("** using frames from  : %s" % args.image_path)
    print("** saving features at : %s" % args.out_path)
    extractor.eval()
    extractor.cuda()
    setting = MODEL_SETTING[args.model]['imagenet']

    for vid in tqdm(os.listdir(args.image_path)):
        tmp_path = os.path.join(args.image_path,vid)
        tmp_loader = Data.DataLoader(VideoFrames(setting,tmp_path,opt.sample_mode),
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=8)
        vfeats = None
        for i, imgs in enumerate(tmp_loader):
            imgs = imgs.cuda()
            tmp_feats,_ = extractor(imgs)
            tmp_feats = tmp_feats.data.cpu()
            if vfeats is None:
                vfeats = tmp_feats
            else:
                vfeats = torch.cat((vfeats,tmp_feats),0)
        np.save(os.path.join(args.out_path,vid[:-4]+'.npy'),vfeats.numpy())



if __name__=="__main__":
    opt = parse_args()
    if opt.model_type == '2d':
        extract_2d_features(opt)