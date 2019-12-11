#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   __init__.py.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/12/4 18:14   Daic       1.0        
'''
import torch
from .models import *
from .loader import get_universal_transform,get_scale,VideoFrames