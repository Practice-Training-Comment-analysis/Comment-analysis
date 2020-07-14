# %%


# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LDA_visualization.py
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/13 9:28   EvanHong      1.0         None
'''
from LDA_related.MyLDA import LDATopicModel

"""
    执行lda2vec.ipnb中的代码
    模型LDA
    功能：训练好后模型数据的可视化
"""
import pandas as pd
from gensim import corpora, models
from LDA_related import MyLDA

import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def get_data(data_path):
    """

    :param data_path: 语料库路径
    :return: dataframe ， 空格分割的文档df
    """
    basic = MyLDA.Basic('../../resources/stopwords.txt', data_path)
    # corpus=[''.join(t for t in (x for x in basic.load_data()))]
    corpus = []
    data = basic.load_data()
    for x in data:
        corpus.append(' '.join(x))
    df = pd.DataFrame(corpus, columns=None)
    print()
    return df


def visualize(data_path):
    """

    :param data_path: 语料库路径
    :return: 无返回，将结果存为HTML，放于lda results目录下
    """
    corpus = get_data('../../resources/data/meidi_comments.txt')

    vectorizer = CountVectorizer()
    print(corpus[0].copy())
    doc_term_matrix = vectorizer.fit_transform(corpus[0])
    # lda_model = LatentDirichletAllocation(n_components=2, random_state=888)
    lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                                          evaluate_every=-1, learning_decay=0.7,
                                          learning_method='batch', learning_offset=10.0,
                                          max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
                                          n_components=10, n_jobs=None, n_topics=None, perp_tol=0.1,
                                          random_state=888, topic_word_prior=None,
                                          total_samples=1000000.0, verbose=0)
    lda_model.fit(doc_term_matrix)



    data = pyLDAvis.sklearn.prepare(lda_model, doc_term_matrix, vectorizer)
    # 让可视化可以在notebook内显示
    pyLDAvis.save_html(data, '/LDA_results/lda_vis.html')

visualize(None)