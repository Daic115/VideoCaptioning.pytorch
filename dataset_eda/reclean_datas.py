#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   reclean_datas.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/12/3 14:05   Daic       1.0        
'''
import os
import json
import pandas as pd
from nltk.tokenize import wordpunct_tokenize

'''
MSVD:
1. lower captions
2. remove bad endings and punctuations
'''