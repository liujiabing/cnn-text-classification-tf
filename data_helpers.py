#!/bin/env python
#_*_coding:utf-8_*_

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import re
import os
import numpy as np
import conf
from collections import defaultdict

WORDVEC = defaultdict()

def filter_text(txt):
    """
    过滤短文本，全英文, {}等
    """
    if len(txt.decode('utf-8')) <= 3:
        return True
    if txt.startswith("&$#@~^@"):
        return True
    if txt.startswith('{') and txt.endswith('}'):
        return True
    return False

def chtnum2num(ustr):
    """
    汉字数字改成数字, 刻度转换
    """
    ustr = re.sub(' ', '', ustr)
    cht2num = {"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "0", "百": "00", "千": "000", "零": "0"}
    for key in cht2num:
        ustr = re.sub(key.decode('utf8'), cht2num[key], ustr.decode('utf8'))
    ustr = re.sub(r'(\d)cm', ur' \1厘米 ', ustr)
    ustr = re.sub(r'(\d)kg', ur' \1千克 ', ustr)
    ustr = re.sub(r'(\d\.\d+)m', ur' \1米 ', ustr)
    return ustr

def clean_str(ustr, remove_dnn_pause = False):
    """
    清理句子, 保留字母,汉字,个别标点符号
    """
    #retstr = re.sub(ur'[^\u4e00-\u9fa5]', '', ustr.decode('utf8'))
    if remove_dnn_pause:
        retstr = re.sub(ur'[^A-Za-z0-9\u4e00-\u9fa5]', '', ustr.decode('utf8')) # remove symbol(dnn_pause)
    else:
        retstr = re.sub(ur'[^A-Za-z0-9,?.\u4e00-\u9fa5]', '', ustr.decode('utf8'))
    return retstr

def cut_words(ustr):
    """
    按字和空格切分
    """
    retstr = ''
    for cht in ustr.decode('utf8'):
        if cht >= u'\u4e00' and cht <= u'\u9fa5' or cht == ',':
            if len(retstr) != 0 and retstr[-1] != ' ':
                retstr += ' ' + cht + ' '
            else:
                retstr += cht + ' '
        else:
            retstr += cht
    return retstr.strip()

def rep_str(ustr):
    """
    替换数字和标点符号
    """
    retstr = re.sub(r' \d+.\d* ', ' dnn_num ', ustr.decode('utf8'))
    retstr = re.sub(r' \d+.\d*[,.?]+ ', ' dnn_num dnn_pause ', retstr.decode('utf8'))
    retstr = re.sub(r' [,.?]+\d+.\d* ', ' dnn_pause dnn_num ', retstr.decode('utf8'))
    retstr = re.sub(r' \d+ ', ' dnn_num ', retstr.decode('utf8'))
    retstr = re.sub(r' \d+[,.?]+ ', ' dnn_num dnn_pause ', retstr.decode('utf8'))
    retstr = re.sub(r' [,.?]+\d+ ', ' dnn_pause dnn_num ', retstr.decode('utf8'))
    retstr = re.sub(r' [,.?]+ ', ' dnn_pause ', retstr.decode('utf8'))
    retstr = re.sub(r' [,.?]+', ' dnn_pause ', retstr.decode('utf8'))
    retstr = re.sub(r'[,.?]+ ', ' dnn_pause ', retstr.decode('utf8'))
    retstr = re.sub(r' [^ ]{30}.+ ', ' ', retstr.decode('utf8'))
    return retstr

def format_text(txt):
    """
    格式化文本
    """
    ustr = re.sub(r'\[.*?\]', '', txt) #表情或者url，TODO.识别出来
    if filter_text(txt):
        return ""
    ustr = ustr.strip().lower()
    ustr = chtnum2num(ustr)
    ustr = str_fw2hw(ustr)
    ustr = clean_str(ustr)
    ustr = cut_words(ustr)
    ustr = rep_str(' ' + ustr + ' ')
    return ustr.strip()

def str_fw2hw(ustr):
    """
    全角转半角
    """
    retstr = ''
    normal = u' ,0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
    wide = u'　，０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ！゛＃＄％＆（）＊＋、ー。／：；〈＝〉？＠［\\］＾＿‘｛｜｝～'
    widemap = dict((x[0], x[1]) for x in zip(wide, normal))
    for cht in ustr.decode("utf8"):
        if cht in widemap.keys():
            retstr += widemap[cht]
        else:
            retstr += cht
    return retstr

def text2mat(text):
    """
    convert text to matrix
    """
    global WORDVEC
    mat = []
    if WORDVEC == defaultdict():
        load_wordvec()
    txt = format_text(text).split()
    txt.reverse()
    for i in range(conf.maxlen):
        if i < len(txt):
            key = txt[i].encode('utf-8')
            if WORDVEC.has_key(key):
                mat.append(WORDVEC[key])
                continue
        mat.append(WORDVEC['dnn_pad'])
    return mat

def load_data_and_labels(data_file):
    """
    Loads data from file, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    lines = open(data_file, 'r').readlines()
    x = []
    for line in lines:
        text = line.split('\t')[0].strip()
        x.append(text2mat(text))

    y_tmp = [ int(_.strip().split('\t')[1]) for _ in lines ]
    y, classes = [], max(y_tmp) + 1
    if classes == 1:
        print >>sys.stderr, 'only one label...'
        #sys.exit(0)
    for c in y_tmp:
        y_i = [0 for _ in range(classes)]
        y_i[c] = 1
        y.append(y_i)
    assert(len(x) == len(y))

    return [np.array(x, float), np.array(y, float)]

def load_wordvec():
    """
    load word vectors from google word2vec project
    """
    lines = open(conf.wordvec).readlines()
    for line in lines[2:]:
        key = line.strip().split()[0]
        value = [ float(_) for _ in line.strip().split()[1:] ]
        WORDVEC.update({key: value})
    WORDVEC.update({'dnn_pad': [ 0.0 for _ in range(int(lines[0].strip().split()[1]))]})
    assert(len(WORDVEC) == int(lines[0].strip().split()[0]))

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
