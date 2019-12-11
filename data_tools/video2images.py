#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   video2images.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/12/4 18:16   Daic       1.0        
'''
import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--out_path', type=str)
    return parser.parse_args()


if __name__=="__main__":
    opt = parse_args()
    input_files = os.listdir(opt.video_path)
    out_path = opt.out_path
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for input_file in input_files:
        video_path = os.path.join(opt.video_path, input_file)
        save_path = os.path.join(out_path, input_file)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        subprocess.call('ffmpeg -i {}/image_%05d.jpg'.format(video_path+' '+save_path),shell=True)