# Data Preparation for Video Captioning
This code implements of data preparation for video captioning based on PyTorch.

### Overview
Data preparation requires three steps:

 1. Video to images: *using `ffmpeg` to get each frame image of the video.*
 2. Images to features: *using pretrained 2d/3d CNN to get their features.*
 3. Preprocess text data to json.
 
Pretrained weight supported almost all types of 2d cnn in [pretrained-models-pytorch].
And 3d cnn only supported Resnet/Resnext in [3d-ResNet].

By the way, the image data generated in the first step is for some visualization experiments.
 If there is no need, it can be deleted after feature extraction(cause it's very large, about
  25GB for MSR-VTT; 6.5GB for MSVD  ).

### Requirements
First, make sure you have `ffmpeg` software on your system. 

If not, using:
`sudo apt-get install ffmpeg`

My python version is 3.6, python2 I have not test yet.
Other pakages:
`pytorch>=1.0.0`

## Step 1
```bash
python video2images.py \ 
--video_path YOUR_VIDEO_PATH \
--out_path SAVE_VIDEO_FRAMES_PATH
```

## Step 2

## Step 3
Preparing yourself or using mine[download]:

```bash
python preprocess_labels.py --  --
```


