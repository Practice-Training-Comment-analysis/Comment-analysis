#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   csv_to_txt.py    
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/15 12:05   EvanHong      1.0         None
'''

import pandas as pd

def comments_csv_to_txt(csv_path, txt_path):
    """
    将csv的评论保存为txt
    :param csv_path:
    :param txt_path:
    :return:
    """

    with open(txt_path, 'w', encoding='utf-8') as fp:
        df = pd.read_csv(open(csv_path, 'r', encoding='utf-8'))
        comments = df['comment']
        t = comments.values.tolist()
        fp.writelines([line + '\n' for line in t])

# comments_csv_to_txt(r'../resources/data/keyword_extract/appearance_comment.csv',r'../resources/data/txts/appearance_comment.txt')