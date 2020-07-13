#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LDA_visualization.py    
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/13 9:28   EvanHong      1.0         None
'''

"""
    执行lda2vec.ipnb中的代码
    模型LDA
    功能：训练好后模型数据的可视化
"""

from lda2vec import preprocess, Corpus
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
import pyLDAvis
try:
    import seaborn
except:
    pass
# 加载训练好的主题-文档模型，这里是查看数据使用。这里需要搞清楚数据的形式，还要去回看这个文件是怎么构成的
npz = np.load(open('D:/my_AI/lda2vec-master/examples/twenty_newsgroups/lda2vec/topics.pyldavis.npz', 'rb'))
# 数据
dat = {k: v for (k, v) in npz.iteritems()}
# 词汇表变成list
dat['vocab'] = dat['vocab'].tolist()

#####################################
##  主题-词汇
#####################################
# 主题个数为10
top_n = 10
# 主题对应10个最相关的词
topic_to_topwords = {}
for j, topic_to_word in enumerate(dat['topic_term_dists']):
    top = np.argsort(topic_to_word)[::-1][:top_n]               # 概率从大到小的下标索引值
    msg = 'Topic %i '  % j
    # 通过list的下标获取关键词
    top_words = [dat['vocab'][i].strip()[:35] for i in top]
    # 数据拼接
    msg += ' '.join(top_words)
    print(msg)
    # 将数据保存到字典里面
    topic_to_topwords[j] = top_words

import warnings
warnings.filterwarnings('ignore')
prepared_data = pyLDAvis.prepare(dat['topic_term_dists'], dat['doc_topic_dists'],
                                 dat['doc_lengths'] * 1.0, dat['vocab'], dat['term_frequency'] * 1.0, mds='tsne')

from sklearn.datasets import fetch_20newsgroups
remove=('headers', 'footers', 'quotes')
texts = fetch_20newsgroups(subset='train', remove=remove).data


##############################################
##  选取一篇文章，确定该文章有哪些主题
##############################################

print(texts[1])
tt = dat['doc_topic_dists'][1]
msg = "{weight:02d}% in topic {topic_id:02d} which has top words {text:s}"
# 遍历这20个主题，观察一下它的权重，权重符合的跳出来
for topic_id, weight in enumerate(dat['doc_topic_dists'][1]):
    if weight > 0.01:
        # 权重符合要求，那么输出该主题下的关联词汇
        text = ', '.join(topic_to_topwords[topic_id])
        print (msg.format(topic_id=topic_id, weight=int(weight * 100.0), text=text))

# plt.bar(np.arange(20), dat['doc_topic_dists'][1])

print(texts[51])
tt = texts[51]
msg = "{weight:02d}% in topic {topic_id:02d} which has top words {text:s}"
for topic_id, weight in enumerate(dat['doc_topic_dists'][51]):
    if weight > 0.01:
        text = ', '.join(topic_to_topwords[topic_id])
        print(msg.format(topic_id=topic_id, weight=int(weight * 100.0), text=text))


# plt.bar(np.arange(20), dat['doc_topic_dists'][51])