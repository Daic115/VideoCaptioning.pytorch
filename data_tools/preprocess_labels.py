#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   preprocess_labels.py
@Desciption     :   None
@Modify Time      @Author    @Version
------------      -------    --------
2019/12/5 0:05   Daic       1.0
'''
import os
import re
import json
import string
import pandas as pd
from collections import Counter
from nltk.tokenize import RegexpTokenizer

'''
clean the MSVD dataset
and
build:
1. tokens
2. word2idx
3. idx2word
4. word_freq_dict
'''
class TrimExceptAscii:
    def __call__(self, sentence):
        s = sentence.decode('ascii', 'ignore').encode('ascii')
        return s

class RemovePunctuation:
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, sentence):
        return self.regex.sub('', sentence)

param_MSVD = {
    'min_freq': 3,
    'max_len': 20,  # > 20 drop caption just in train-set
}
param_MSRVTT = {
    'min_freq': 3,
    'max_len': 20,  # > 20 drop caption just in train-set
}

TOKENIZER = RegexpTokenizer(r'\w+')
ASCII_PROCESS = TrimExceptAscii()
PUNCTUATION_REMOVER = RemovePunctuation()

def process_and_tokenize(sentence):
    s = sentence.lower()
    s = ASCII_PROCESS(s)
    s = PUNCTUATION_REMOVER(s)
    t = TOKENIZER.tokenize(s)
    return s , t

def preprocess_MSVD_label(dftr, dfva, dfte):
    datasets = {'train': {}, 'validate': {}, 'test': {}}

    def apply_clip_id(x):
        x['clip_id'] = x['VideoID'] + '_' + str(x['Start']) + '_' + str(x['End'])
        return x

    dftr = dftr.apply(apply_clip_id, axis=1)
    dfva = dfva.apply(apply_clip_id, axis=1)
    dfte = dfte.apply(apply_clip_id, axis=1)

    vidset = set(dftr['clip_id'])
    for vid in vidset:
        datasets['train'][vid] = {'info': [], 'sents': [], 'sent_ids': [], 'tokens': []}
    vidset = set(dfva['clip_id'])
    for vid in vidset:
        datasets['validate'][vid] = {'info': [], 'sents': [], 'sent_ids': [], 'tokens': []}
    vidset = set(dfte['clip_id'])
    for vid in vidset:
        datasets['test'][vid] = {'info': [], 'sents': [], 'sent_ids': [], 'tokens': []}

    sp = 'train'
    for ix in range(len(dftr)):
        tmp_vid = dftr.iloc[ix]['clip_id']
        tmp_sent,tmp_token = process_and_tokenize(dftr.iloc[ix]['Description'])
        # tmp_sent = dftr.iloc[ix]['Description'].lower()  # STRING
        # tmp_token = TOKENIZER.tokenize(tmp_sent)  # LIST
        datasets[sp][tmp_vid]['sents'].append(tmp_sent)
        datasets[sp][tmp_vid]['tokens'].append(tmp_token)

    sp = 'validate'
    for ix in range(len(dfva)):
        tmp_vid = dfva.iloc[ix]['clip_id']
        tmp_sent, tmp_token = process_and_tokenize(dfva.iloc[ix]['Description'])
        datasets[sp][tmp_vid]['sents'].append(tmp_sent)
        datasets[sp][tmp_vid]['tokens'].append(tmp_token)

    sp = 'test'
    for ix in range(len(dfte)):
        tmp_vid = dfte.iloc[ix]['clip_id']
        tmp_sent, tmp_token = process_and_tokenize(dfte.iloc[ix]['Description'])
        datasets[sp][tmp_vid]['sents'].append(tmp_sent)
        datasets[sp][tmp_vid]['tokens'].append(tmp_token)

    # label just for train split:
    ## counting:
    counts_tr = {}
    for vid in datasets['train'].keys():
        for sent in datasets['train'][vid]['tokens']:
            for w in sent:
                counts_tr[w] = counts_tr.get(w, 0) + 1

    ## label index
    word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    idx = 3
    for w in counts_tr.keys():
        if counts_tr[w] > param_MSRVTT['min_freq']:
            word2idx[w] = idx
            idx += 1
    idx2word = {v: k for k, v in word2idx.items()}

    return {'data': datasets,
            'word2idx': word2idx,
            'idx2word': idx2word,
            'word_freq': counts_tr,
            'param':param_MSVD}


def preprocess_MSRVTT_labels(tv_infos, te_infos):
    datasets = {'train': {}, 'validate': {}, 'test': {}}
    for v in tv_infos['videos']:
        datasets[v['split']][v['video_id']] = {'info': v, 'sents': [], 'sent_ids': [], 'tokens': []}
    for v in te_infos['videos']:
        datasets[v['split']][v['video_id']] = {'info': v, 'sents': [], 'sent_ids': [], 'tokens': []}

    for sp in ['train', 'validate']:
        vid_set = set(datasets[sp].keys())
        for sent in tv_infos['sentences']:
            if sent['video_id'] in vid_set:
                tmp_sent = sent['caption'].lower()  # STRING
                tmp_token = TOKENIZER.tokenize(tmp_sent)  # LIST
                datasets[sp][sent['video_id']]['sents'].append(tmp_sent)
                datasets[sp][sent['video_id']]['sent_ids'].append(sent['sen_id'])
                datasets[sp][sent['video_id']]['tokens'].append(tmp_token)
    sp = 'test'
    for sent in te_infos['sentences']:
        tmp_sent = sent['caption'].lower()  # STRING
        tmp_token = TOKENIZER.tokenize(tmp_sent)  # LIST
        datasets[sp][sent['video_id']]['sents'].append(tmp_sent)
        datasets[sp][sent['video_id']]['sent_ids'].append(sent['sen_id'])
        datasets[sp][sent['video_id']]['tokens'].append(tmp_token)

    # label just for train split:
    ## counting:
    counts_tr = {}
    for vid in datasets['train'].keys():
        for sent in datasets['train'][vid]['tokens']:
            for w in sent:
                counts_tr[w] = counts_tr.get(w, 0) + 1

    ## label index
    word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    idx = 3
    for w in counts_tr.keys():
        if counts_tr[w] > param_MSRVTT['min_freq']:
            word2idx[w] = idx
            idx += 1
    idx2word = {v: k for k, v in word2idx.items()}

    return {'data': datasets,
            'word2idx': word2idx,
            'idx2word': idx2word,
            'word_freq': counts_tr,
            'param':param_MSRVTT}


if __name__ == "__main__":
    print("processing msvd labels...")
    msvd_df_tr = pd.read_csv("./split_datas/msvd/train.csv")
    msvd_df_va = pd.read_csv("./split_datas/msvd/val.csv")
    msvd_df_te = pd.read_csv("./split_datas/msvd/test.csv")
    msvd_data = preprocess_MSVD_label(msvd_df_tr,msvd_df_va,msvd_df_te)
    json.dump(msvd_data,open('./split_datas/MSVD_data0.json','w'))

    print("processing msr-vtt labels...")
    msratt_train_val = json.load(open('./split_datas/msrvtt/train_val_videodatainfo.json'))
    msratt_test = json.load(open('./split_datas/msrvtt/test_videodatainfo.json'))
    msrvtt_data=preprocess_MSRVTT_labels(msratt_train_val,msratt_test)
    json.dump(msrvtt_data, open('./split_datas/MSRVTT_data0.json', 'w'))
