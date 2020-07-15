#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   classify_by_emotion.py    
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/13 23:18   EvanHong      1.0         None
'''

# %%

import pandas as pd

# %%

import os


def get_doc_path(root_path):
    """
    获取每个csv评论文件的路径
    :return: list
    """
    dirs = []
    # 打开文件
    for dir in os.listdir(root_path)[1:]:
        dirs.append(str(root_path + '/' + dir))
    # 输出所有文件和文件夹
    return dirs


def get_doc():
    """
    获取单个CSV文件（即单个商品）的评论内容，并存入一个列表
    :param doc_path:
    :return: list-> ['str1','str2',...]
    """
    root_path_to_read = '../resources/data/classified_comment'
    root_path_to_write = '../resources/data/classified_comment_by_emotion'
    doc_paths_read = get_doc_path(root_path_to_read)
    filenames = os.listdir(root_path_to_read)[1:]
    # doc_paths_write=get_doc_path(root_path_to_write)

    # 遍历文件夹下csv文件，分别生成积极和消极，存入新的文件夹
    for doc_path in doc_paths_read:
        data = pd.read_csv(open(doc_path, encoding="utf-8"))

        # %%

        # 选定分类方法（星级或情感分数）以及评判线
        goodlist, badlist = classify_byscore(4, 3, data)
        for filename in filenames:

            gooddf = pd.DataFrame(goodlist['comment'])
            baddf = pd.DataFrame(badlist['comment'])

            gooddf.to_csv(root_path_to_write+'/' + filename[:-4] + '_positive.csv',index=False, header=True,encoding='utf-8')
            baddf.to_csv(root_path_to_write +'/' + filename[:-4] + '_negative.csv', index=False, header=True,encoding='utf-8')


def classify_byscore(goodscore, badscore, df):
    goodlist = df[df["score"] >= goodscore]
    badlist = df[df["score"] <= badscore]
    return goodlist, badlist


def classify_bysentiment(goodsentiment, badsentiment, df):
    goodlist = df[df["sentiment"] >= goodsentiment]
    badlist = df[df["sentiment"] <= badsentiment]
    return goodlist, badlist


# %%


# %%

# gooddf=pd.DataFrame(goodlist['comment'])
# baddf=pd.DataFrame(badlist['comment'])
#
# %%

# gooddf.to_csv(filename[:-4]+"_正面.csv",index=False,header=True)
# baddf.to_csv(filename[:-4]+"_负面.csv",index=False,header=True)

# %%
get_doc()
