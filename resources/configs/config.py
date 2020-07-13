#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/12 15:40   EvanHong      1.0         None
'''

#crawler config
HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
    'accept': 'text / html, application / xhtml + xml, application / xml;q = 0.9, image / webp, image / apng, * / *;q = 0.8, application / signed - exchange;v = b3;q = 0.9',
    'accept - encoding': 'gzip, deflate, br',
    'accept - language': 'en - US, en;q = 0.9, zh - CN;q = 0.8, zh;q = 0.7',
    'cache - control': 'max - age = 0'
}

# LDA config
STOP_WORD_PATH = r'../../Comment_analysis/Wordcloud/stopword.txt'
CORPUS_PATH = r'../../resources/data/meidi_comments.txt'
KEYWORD_NUM=5
NUM_OF_TOPICS=2