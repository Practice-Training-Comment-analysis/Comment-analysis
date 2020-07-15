#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py    
@Contact :   h939778128@gmail.com
@License :   No license


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/13 15:37   EvanHong      1.0         把csv文件写成txt，以便lda
'''

import pandas as pd
import numpy as np

import os


def get_doc_path(root_path):
    """
    获取每个csv评论文件的路径
    :return: list
    """
    dirs=[]
    # 打开文件
    for dir in os.listdir(root_path)[1:]:
        dirs.append(str(root_path+ '/'+dir))
    # 输出所有文件和文件夹
    return dirs

def get_doc():
    """
    获取单个CSV文件（即单个商品）的评论内容，并存入一个列表
    :param doc_path:
    :return: list-> ['str1','str2',...]
    """
    root_path = '../../resources/data/classified_comment'
    doc_paths = get_doc_path(root_path)

    with open('../../resources/data/meidi_comments.txt', 'w', encoding='utf-8') as fp:
        for doc_path in doc_paths:
            df = pd.read_csv(open(doc_path, 'r', encoding='utf-8'))
            comments = df['comment']
            t = comments.values.tolist()
            fp.writelines([line + '\n' for line in t])






