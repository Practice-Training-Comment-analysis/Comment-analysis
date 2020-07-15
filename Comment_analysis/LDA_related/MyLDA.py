# -*- encoding: utf-8 -*-
import logging
import math

from gensim.models import CoherenceModel

from configs import config
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools
import logging
import inspect



class Basic(object):

    def __init__(self, stop_word_path, corpus_path):
        self.stop_word_path = stop_word_path
        self.corpus_path = corpus_path
        self.logger = logging.getLogger('crawler logger')

    def log(self, e=None, message=None):
        """
        日志记录函数，可以记录报错信息e，也可以记录message
        :param e: exception object
        :param message: str
        :return:
        """
        # 获取当前运行函数信息
        func = inspect.currentframe().f_back.f_code

        # 不能在函数里面声明logger，否则会导致它一直留存在内存且越累积越多，最后一个信息有无数个logger在写
        # 解决，在文件开头声明一个文件全局logger，此log函数调用那个logger就行
        # 好像没解决，哭
        # logger = logging.getLogger()
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("../../resources/logs/log.txt", encoding='utf-8')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
            self.logger.addHandler(console)
        if message is not None:
            self.logger.info(message)
        # logger.debug("Do something")
        # "Automatically log the current function details."
        # Get the previous frame in the stack, otherwise it would
        # be this function!!!
        if e is not None:
            # Dump the message + the name of this function to the log.
            self.logger.error("%s:  %s  in  %s :  %i" % (
                e.__str__(),
                func.co_name,
                func.co_filename,
                func.co_firstlineno
            ))
            print(e.__str__(),
                  func.co_name,
                  func.co_filename,
                  func.co_firstlineno)
            self.logger.warning("Something maybe fail.")
        # logger.info("Finish")

    def get_stop_words(self, stop_word_path):
        stop_words = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8').readlines()]
        return stop_words

    def seg_to_list(self, sentence, ifPosseg=False):
        """
        定义分词方法，jieba
        :param sentence:
        :param ifPosseg:
        :return:
        """
        if ifPosseg:
            return psg.cut(sentence)
        else:
            return jieba.cut(sentence)

    def filt_stpwds(self, seg_list, ifPosseg=False):
        """

        :param stop_word_path:
        :param seg_list:
        :param ifPosseg:
        :return:
        """
        stop_words = self.get_stop_words(self.stop_word_path)
        res = []
        # 不进行磁性过滤，则词性为n，表示全部保留
        for seg in seg_list:
            if not ifPosseg:
                word = seg
                flag = 'n'
                # print(word)
            else:
                word = seg.word
                flag = seg.flag
            if not flag.startswith('n'):
                continue
            else:
                if seg not in stop_words and len(word) > 1:
                    res.append(word)
        return res

    def load_data(self, ifPosseg=False):
        """
        加载数据
        :param corpus_path:
        :param ifPosseg: 是否词性标注
        :return: word_list segment
        """

        word_list = []
        corpus=open(self.corpus_path, encoding='utf-8').readlines()
        if corpus:

            for line in corpus:
                content = line.strip()
                # 分词
                seg_content_list = self.seg_to_list(content, ifPosseg)
                filted_content_list = self.filt_stpwds(seg_content_list)
                word_list.append(filted_content_list)
        return word_list

    def train_idf(self, doc_list):
        """
        idf值统计方法
        每个词是否在各个文档中的出现
        :return:
        """
        idf_dic = {}  # 总文档数据

        tt_count = idf_dic.__sizeof__()  # 总文档数（一行一个文档

        # 每个词出现的文档数
        for doc in doc_list:
            for word in doc:
                """0 is the default value"""
                idf_dic[word] = idf_dic.get(word, 0) + 1

        # 按公式转换为idf值，分母加1进行平滑处理(防止v==0时log负无穷
        for k, v in idf_dic.items():
            idf_dic[k] = math.log(tt_count / (1.0 + v))

        # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
        default_idf = math.log(tt_count / (1.0))
        return idf_dic, default_idf


def cmp(e1, e2):
    """
    取top k个值
    :param e1:
    :param e2:
    :return:
    """
    import numpy as np
    res = np.sign(e1[1] - e2[1])  # 取得正负号The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``
    if res != 0:
        return res
    else:
        # 词频/tfidf相同，比较两个字符串的顺序
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


class TfIdf(object):
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        """

        :param idf_dic: 训练好的字典
        :param default_idf: 默认idf
        :param word_list: 待提取文本
        :param keyword_num: 关键词数量
        """
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0) + 1

        tt_count = len(self.word_list)  # 文本中总共的词数
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    def get_tfidf_dic(self):
        """
        生成 tfidf_dic
        :return:
        """
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)
            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(),
                           key=functools.cmp_to_key(cmp),
                           reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')

        return tfidf_dic


class LDATopicModel(object):
    """
    主题模型
    """

    def __init__(self, doc_list, keyword_num, model_type='LDA',if_load_model=False, model_path=None,model_name='temp_model', num_topics=4):
        """

        :param doc_list:
        :param keyword_num:
        :param model_type: 具体模型（LSI、LDA）
        :param num_topics: 主题数量
        """
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间，其中包含了所有的词
        self.dictionary = corpora.Dictionary(doc_list)

        # 使用BOW模型向量化
        # 建立每一个词的键值对，当下次再次出现这个词的时候，仍旧使用这个键，构成方式： [[（键，在此文档中的频率）,（键，在此文档中的频率）],[（键，在此文档中的频率）,（键，在此文档中的频率）]]
        self.corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # print(corpus)
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(self.corpus, dictionary=self.dictionary)
        self.corpus_tfidf = self.tfidf_model[self.corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if if_load_model:
            self.model = models.LdaModel.load(model_path)
        else:
            if model_path is None:
                if model_type == 'LSI':
                    self.model = self.train_lsi()
                    if model_name is None:
                        self.model.save('LDA_related/temp_lsi_model.model')
                    else:
                        self.model.save('LDA_related/{}.model'.format(model_name))
                else:
                    self.model = self.train_lda()
                    if model_name is None:
                        self.model.save('LDA_related/temp_lda_model.model')
                    else:
                        self.model.save('LDA_related/{}.model'.format(model_name))
            else:
                if model_type == 'LSI':
                    self.model = self.train_lsi()
                    self.model.save(model_path)

                else:
                    self.model = self.train_lda()
                    self.model.save(model_path)


        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def get_model(self):
        return self.model

    def printer(self):
        print(self.dictionary)
        print(self.tfidf_model)
        print(self.corpus_tfidf)

    def train_lsi(self):
        """

        :return:
        """
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}

        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    def get_topics(self):
        return self.model.get_topics()

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim
        res = []
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
            res.append(k)
        print()
        return res

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list

    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def LDA_visualization(self):
        pass



def tfidf_extract(word_list, basic, pos=False, keyword_num=10):
    doc_list = Basic.load_data(basic, ifPosseg=pos)
    idf_dic, default_idf = Basic.train_idf(basic, doc_list)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf_dic()


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/ ", end='')


def topic_extract(word_list, model, basic, pos=False, keyword_num=10, num_topics=2,model_path=None,model_name=None,load_model=False):
    doc_list = basic.load_data(ifPosseg=pos)

    topic_model = LDATopicModel(doc_list,if_load_model=load_model, keyword_num=keyword_num,
                                model_type=model, num_topics=num_topics,model_path=model_path,model_name=model_name)

    # print(topic_model.model.show_topics(formatted=False))
    return topic_model.get_simword(word_list), topic_model


def show_topic_words(texts,keyword_num=config.KEYWORD_NUM,num_topics=config.NUM_OF_TOPICS):
    """
    获取与text内容最接近的关键词
    :param text:
    :return:
    """
    pos = False

    #
    basic = Basic(config.STOP_WORD_PATH, config.CORPUS_PATH)
    if_load_model=False
    cnt = 0
    for text in texts:
        seg_list = basic.seg_to_list(sentence=str(text), ifPosseg=pos)
        filter_list = basic.filt_stpwds(seg_list, pos)

        print('新文档:{}\n'.format(cnt), text)

        # 获取与text内容最接近的关键词
        extracted_topics, lda_model = topic_extract(filter_list, 'LDA', basic, pos, keyword_num=keyword_num,
                                                    num_topics=num_topics,load_model=if_load_model,
                                                    model_path='../LDA_related/temp_lda_model.model')
        print(extracted_topics)
        model = lda_model.get_model()

        # 转换成词袋
        bow = model.id2word.doc2bow(filter_list)
        doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)

        topic_list = ['外形外观', '耗能情况', '恒温效果', '噪音大小', '安装服务']

        doc_topic = [(topic_list[i[0]], i[1]) for i in doc_topics]
        word_topic = [(lda_model.dictionary.id2token[i[0]], i[1]) for i in word_topics]
        phi_value = [(lda_model.dictionary.id2token[i[0]], i[1]) for i in phi_values]
        print(" 文档主题:", doc_topic)
        print(" 词汇主题:", word_topic)
        print(" Phi:", phi_value)
        print("============================\n")

        cnt += 1
        if_load_model=True




# def set_corpus():


if __name__ == '__main__':
    """
    现阶段需求分析
    对'''一个商品'''的正负情感评论
    """
    text = [['出水很快，外形美观，安装服务好\n质量很差'], ['质量很好，外形外观很漂亮，喜欢这个设计']]
    show_topic_words(text)


