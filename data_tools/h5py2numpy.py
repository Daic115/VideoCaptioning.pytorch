#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   h5py2numpy.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/12/8 16:41   Daic       1.0        
'''
import os
import torch
import argparse
import h5py
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path',    type=str,                           help="The directory path of videos.")
    parser.add_argument('--out_path',   type=str,                           help="The output path of extracted feature.")
    return parser.parse_args()

def main(opt):
    fin = h5py.File(opt.h5_path, 'r')
    if not os.path.isdir(opt.out_path):
        os.mkdir(opt.out_path)

    for vid in fin.keys():
        feats = fin[vid].value
        np.save(os.path.join(opt.out_path,vid+'.npy'),feats)

if __name__=="__main__":
    opt = parse_args()
    main(opt)