#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   csv_to_txt.py    
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/15 12:05   EvanHong      1.0         将存储于csv的列评论转换为txt，为pyLDAvis输入提供语料
'''

import pandas as pd
from stmclf.classify_by_emotion import get_csv_path
import re
import jieba
import jieba.posseg as pseg
def comment_csv_to_txt(csv_path, txt_path,stop_words=None):
    """
    将一个csv的评论保存为txt
    :param csv_path:
    :param txt_path:
    :return:
    """

    with open(txt_path, 'w', encoding='utf-8') as fp:
        df = pd.read_csv(open(csv_path, 'r', encoding='utf-8'))
        comments = df['comment']
        t = comments.values.tolist()
        fp.writelines([str(line) + '\n' for line in t])



def comments_csv_to_txt(root_path_to_read, root_path_to_write):
    """
    将目录下所有csv的评论保存为txt
    :param root_path_to_read:
    :param root_path_to_write:
    :return:
    """
    # 加载停用词
    stop_words = []
    file = open('../../resources/stopwords.txt', 'r', encoding='utf-8').readlines()  # 自定义去除词库
    for each_line in file:
        each_line = each_line.strip('\n')
        stop_words.append(each_line)
    doc_paths_read = get_csv_path(root_path_to_read)

    # 遍历文件夹下csv文件，分别生成积极和消极，存入新的文件夹
    for i in range(len(doc_paths_read)):
        filename = re.findall('/([^/]*).csv', doc_paths_read[i])
        if filename:
            comment_csv_to_txt(doc_paths_read[i], root_path_to_write + '/' + filename[0] + '.txt',stop_words)


comments_csv_to_txt('../../resources/data/meidi_yearly_comment', '../../resources/data/txts/meidi_yearly_comment')
