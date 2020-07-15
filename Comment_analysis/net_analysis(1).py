import re  # 正则表达式库
import jieba  # 分词
import collections  # 词频统计库
import numpy as np
import pandas as pd
import networkx as nx  # 复杂网络分析库
import matplotlib.pyplot as plt


# print(dir(nx.layout))
def networkx_analysis(dir='美的（Midea）JSQ22-L1(Y)_comment_正面.csv'):
    '''
    输入待分析的csv文件路径
    输出networkx网络分析结果
    '''
    # 初始化
    num = 40
    G = nx.Graph()
    plt.figure(figsize=(20, 14))
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    # 读取文件
    fn = pd.read_csv(dir, encoding='utf-8', engine='python')  # 打开文件
    string_data = fn['comment']  # 读出评论文件
    print(len(string_data))
    file = open('meidi.txt', 'w')

    string_all = ''
    # 文本预处理
    pattern = re.compile(u'\t|。|，|：|；|！|）|（|？|、|“|”')  # 定义正则表达式匹配模式
    for i in range(len(string_data)):
        string_all += re.sub(pattern, '', string_data[i])
        file.write(string_data[i] + '\n')

    # string_data = re.sub(pattern, '', string_data) # 将符合模式的字符去除

    # 文本分词
    seg_list_exact = jieba.cut(string_all, cut_all=False)  # 精确模式分词
    object_list = []
    stop_words = []
    file = open('stopwords.txt', 'r', encoding='utf-8').readlines()  # 自定义去除词库
    for each_line in file:
        each_line = each_line.strip('\n')
        stop_words.append(each_line)
    print(stop_words)

    for word in seg_list_exact:  # 循环读出每个分词
        if word not in stop_words:  # 如果不在去除词库中
            object_list.append(word)  # 分词追加到列表

    print('object_list的个数：', len(object_list))
    # 词频统计
    word_counts = collections.Counter(object_list)  # 对分词做词频统计
    word_counts_top = word_counts.most_common(num)  # 获取最高频的词
    word = pd.DataFrame(word_counts_top, columns=['关键词', '次数'])
    print(word)

    # word_T = pd.DataFrame(word.values.T)
    # word_T.to_excel('word_T.xls')
    # print('==='*30)
    # print(word_T)

    net = pd.DataFrame(np.mat(np.zeros((num, num))), columns=word.iloc[:, 0])

    print('===' * 30)
    # print(net)

    k = 0
    object_list2 = []
    # 构建语义关联矩阵
    for i in range(len(string_data)):
        seg_list_exact = jieba.cut(string_data[i], cut_all=False)  # 精确模式分词
        object_list2 = []
        for words in seg_list_exact:  # 循环读出每个分词
            if words not in stop_words:  # 如果不在去除词库中
                object_list2.append(words)  # 分词追加到列表
        # print(object_list2)

        word_counts2 = collections.Counter(object_list2)
        word_counts_top2 = word_counts2.most_common(num)  # 获取该段最高频的词
        word2 = pd.DataFrame(word_counts_top2)
        word2_T = pd.DataFrame(word2.values.T, columns=word2.iloc[:, 0])

        # word2_T.to_excel('word2_T.xls')
        # print(word2_T)

        relation = list(0 for x in range(num))
        # 查看该段最高频的词是否在总的最高频的词列表中
        for j in range(num):
            for p in range(len(word2)):
                if word.iloc[j, 0] == word2.iloc[p, 0]:
                    relation[j] = 1
                    break
                # 对于同段落内出现的最高频词，根据其出现次数加到语义关联矩阵的相应位置
        for j in range(num):
            if relation[j] == 1:
                for q in range(num):
                    if relation[q] == 1:
                        net.iloc[j, q] = net.iloc[j, q] + word2_T.loc[1, word.iloc[q, 0]]
                        net.iloc[q, j] = net.iloc[j, q] + word2_T.loc[1, word.iloc[q, 0]]
    net.to_excel('net.xls')
    print(net)
    # 处理最后一段内容，完成语义关联矩阵的构建
    max_weight = net.get_values().max()
    # 数据归一化
    for i in range(num):
        for j in range(num):
            net.iloc[i, j] = net.iloc[i, j] / max_weight
    n = len(word)
    #         # 边的起点，终点，权重
    for i in range(n):
        for j in range(i, n):
            G.add_weighted_edges_from([(word.iloc[i, 0], word.iloc[j, 0], net.iloc[i, j])])
    nx.draw_networkx(G,
                     pos=nx.circular_layout(G),
                     #                 根据权重大小设置线的粗细,可以自行调节线条的粗细，调节边框的颜色，可以调节图的布局
                     width=[float(v['weight'] * 3) for (r, c, v) in G.edges(data=True)],
                     edge_color='orange',
                     #                根据出现的次数，设置点的大小
                     node_size=[float(net.iloc[i, i] * 2000) for i in np.arange(20)],
                     node_color='#87CEEB',
                     font_size=15,
                     font_weight='1000',
                     )

    plt.axis('off')
    plt.show(G)
