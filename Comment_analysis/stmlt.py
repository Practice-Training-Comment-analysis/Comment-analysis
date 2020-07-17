#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   stmlt.py    
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/17 10:27   EvanHong      1.0         None
'''

import streamlit as st

from PIL import Image
import configs.config as cf



def show_image(image_name, title):
    """

    :param image_name: str
    :return:
    """
    st.image(Image.open(cf.PICTURE_ROOT + image_name), caption=title,use_column_width=True)


# crawler
def show_crawler():
    st.markdown('***')

    st.subheader('数据爬取')

    st.markdown('''
    **数据采集和抽取**
    
    
    - Pymysql完成与数据库的交互
    
    - 用RE、requests、beautifulsoup、lxml对html中的内容进行提取。分别比较了数据的
    
    - 用pandas对json中的数据进行处理
    
    ''')
    if st.button('show code detail'):
        st.code('''def info_to_file():
        """
        将网页信息存入数据库/表
        :return:
        """
        global connection
        try:
            # 连接数据库
            connection = pymysql.connect(host='localhost',
                                         port=3306,
                                         user='root',
                                         password='Qazwsxedcrfv0957',
                                         db='test',
                                         charset='utf8',
                                         cursorclass=pymysql.cursors.DictCursor
    
                                         )
            connection.autocommit(True)
    
            # 写入数据库
            with open('../../resources/data/haier_product_id.csv', 'r', encoding='utf-8') as csvFile:
                reader = csv.reader(csvFile)
    
                for line in reader:
                    for id in line:
                        print('正在检查是否重复' + id)
    
                        # 检查是否重复录入id
                        ids = []
                        with connection.cursor() as cursor:
                            # 创建一条新的记录
                            sql = "select spider.product_info.product_id from spider.product_info"
                            cursor.execute(sql)
                            id_exist = cursor.fetchall()
                            for i in range(len(id_exist)):
                                ids.append(id_exist[i].get('product_id'))
                        if int(id) in ids:
                            print('id重复')
                            continue
    
                        # 没有重复录入，开始爬取信息
                        brand, model, price, goodCount, generalCount, poorCount = get_basic_product_info(id)
                        # 如果信息为空，则不录入
                        if (brand is None or model is None or price is None) or (
                                goodCount is None or generalCount is None or poorCount is None):
                            break
                        print('基本信息开始录入' + id)
                        log(None, '基本信息开始录入' + id)
                        with connection.cursor() as cursor:
                            # 创建一条新的记录
                            sql = "INSERT INTO `spider`.product_info(product_id,brand,model,price,good_count,general_count,poor_count) VALUES (%s,%s, %s,%s,%s, %s,%s)"
                            cursor.execute(sql, (int(id), brand, model, price, goodCount, generalCount, poorCount))
                        log(None, '基本信息已录入' + id)
                        print('基本信息已录入' + id)
    
                        # 连接完数据库并不会自动提交，所以需要手动 commit 你的改动
                        connection.commit()
                        time.sleep(2)
    
                        # write comment related information into table comments
                        # 不同的评论种类，其中包含带图评论等
                        for score in range(7):
                            # 如果有一个数量为空，则默认最大读取页数为100
                            if generalCount is None or goodCount is None or poorCount is None:
                                page_num = 100
                            else:  # 否则计算最大页数
                                page_num = int((goodCount + generalCount + poorCount) / 10)
                            get_comments_and_to_file(id, page_num, (score + 1))
    
    ''')
    show_image('database.png', '数据库截图')

# LDA
def show_LDA_visualization():
    st.markdown('***')

    st.subheader('LDA主题分析可视化')
    # with st.spinner(text='In progress...'):
    #     time.sleep(5)
    # st.success('done')

    st.markdown("""
    
    - 构建相应的语料库
    
    - 使用sklearn库进行模型训练
    
    - 利用pyLDAvis库将模型可视化
    """)

    # st.image(load_image('../resources/LDA_related/pictures/1.png'), caption='test',
    #        use_column_width=True)
    st.markdown('**进入以下超链接可直接进行互动**')
    st.markdown('>更改lambda值（0-1）可以查看每个主题最具特点的词-每个主题最主要的词（类似于tf idf）')
    display_type = st.selectbox('=>请选择一个选项<=',['美的品牌评论情感分类LDA结果', '美的品牌评论随时间变化LDA结果', '美的品牌与其他品牌热水器在各方面评论对比LDA结果'])
    if display_type == '美的品牌评论情感分类LDA结果':
        st.markdown("""
            **[美的正面评价LDA主题分析结果](https://dyf-2316.github.io/HYF_LDA_results/meidi_summary/美的（Midea）_正面.html)**\n
            **[美的负面评价LDA主题分析结果](https://dyf-2316.github.io/HYF_LDA_results/meidi_summary/美的（Midea）_负面.html)**\n\n
            
        """
        )
        st.markdown("""
        
            `从“美的正面评论总览”和“美的负面评论总览”我们发现，光是从数量上来看的话，安装服务是顾客最在意的点。     
            但是当我们选择了最主要的主题之后可以发现，顾客在一个较为满意的购物体验中，安装的重要性会适当下降,
            反而热水器有关的外观、性能会给他们留下更深刻的印象。
            但是对于差评主要主题，安装、服务、售后、费用等关键词是较为主要的几个点`
        """)
        show_image('meidi_pos.png','美的正面评论总览')
        show_image('meidi_neg.png','美的负面评论总览')
        show_image('meidi_positive.jpg','美的正面评论最主要主题内容')
        show_image('meidi_negative.jpg','美的负面评论最主要主题内容')


    elif display_type == '美的品牌评论随时间变化LDA结果':

        st.markdown('''
            **[2017](https://dyf-2316.github.io/HYF_LDA_results/meidi_yearly_comment/spider_meidi_comments2017.html)**\n
            **[2018](https://dyf-2316.github.io/HYF_LDA_results/meidi_yearly_comment/spider_meidi_comments2018.html)**\n
            **[2019](https://dyf-2316.github.io/HYF_LDA_results/meidi_yearly_comment/spider_meidi_comments2019.html)**\n
            **[2020](https://dyf-2316.github.io/HYF_LDA_results/meidi_yearly_comment/spider_meidi_comments2020.html)**\n

            `从中我们可以看到，在2018年人们的评论主题较大可能为品牌，但是到了2019/2020年，
            人们更加乐于评论外形外观以及性能方面的话题，说明美的在这段时间可能对产品进行了外观上的大升级，
            大大增强了产品竞争力`
        ''')
        show_image('2017.png','美的2017年热水器评论主题分析')
        show_image('2018.png','美的2018年热水器评论主题分析')
        show_image('2019.png','美的2019年热水器评论主题分析')
        show_image('2020.png','美的2020年热水器评论主题分析')


    elif display_type == '美的品牌与其他品牌热水器在各方面评论对比LDA结果':
        st.markdown('''
            `通过对tag进行筛选后我们可以观察到各个方面各品牌产品的特点`
        ''')
        feature = st.radio('请选择一方面进行对比', ['安装服务', '外形外观','耗能情况','恒温效果','噪音大小','出水速度'])
        if feature == '安装服务':
            st.markdown('*** ')
            st.markdown('安装服务')
            st.markdown("""
                `美的产品相较于别的产品在安装服务的及时性以及速度方面评分较高，并且在服务质量、专业性方面和其他品牌均持平`
            """)
            st.markdown(
            

                '''
                            **[美的](https://dyf-2316.github.io/HYF_LDA_results/tag/美的（Midea）安装服务_comment.html)**\n
                            **[海尔](https://dyf-2316.github.io/HYF_LDA_results/tag/海尔（Haier）安装服务_comment.html)**\n
                            **[史密斯](https://dyf-2316.github.io/HYF_LDA_results/tag/史密斯（A.O.S安装服务_comment.html)**\n
                
                        ''')

            show_image('美的安装.png','美的安装')
            show_image('海尔安装.png','海尔安装')
            show_image('史密斯安装.png','史密斯安装')


        elif feature == '外形外观':
            st.markdown('*** ')
            st.markdown('外形外观')
            st.markdown("""
                `美的的产品在外观方面并没有特别突出的特点，不像海尔热水器有一个较为显著的特点是简介，史密斯较为上档次`
            """)
            st.markdown('''
                        **[美的](https://dyf-2316.github.io/HYF_LDA_results/tag/美的（Midea）外形外观_comment.html)**\n
                        **[海尔](https://dyf-2316.github.io/HYF_LDA_results/tag/海尔（Haier）外形外观_comment.html)**\n
                        **[史密斯](https://dyf-2316.github.io/HYF_LDA_results/tag/史密斯（A.O.S外形外观_comment.html)**\n

                    ''')

        elif feature == '耗能情况':
            st.markdown('*** ')
            st.markdown('耗能情况')
            st.markdown("""
                `美的产品可能有较为清楚的能耗标识，因此美的评论主题关键词中“清楚”“知道”排名靠前`
            """)
            st.markdown('''
                        **[美的](https://dyf-2316.github.io/HYF_LDA_results/tag/美的（Midea）耗能情况_comment.html)**\n
                        **[海尔](https://dyf-2316.github.io/HYF_LDA_results/tag/海尔（Haier）耗能情况_comment.html)**\n
                        **[史密斯](https://dyf-2316.github.io/HYF_LDA_results/tag/史密斯（A.O.S耗能情况_comment.html)**\n

                    ''')

        elif feature == '恒温效果':
            st.markdown('*** ')
            st.markdown('恒温效果')
            st.markdown("""
                `美的产品在此方面没有特别突出的点`
            """)
            st.markdown('''
                        **[美的](https://dyf-2316.github.io/HYF_LDA_results/tag/美的（Midea）恒温效果_comment.html)**\n
                        **[海尔](https://dyf-2316.github.io/HYF_LDA_results/tag/海尔（Haier）恒温效果_comment.html)**\n
                        **[史密斯](https://dyf-2316.github.io/HYF_LDA_results/tag/史密斯（A.O.S恒温效果_comment.html)**\n

                    ''')

        elif feature == '噪音大小':
            st.markdown('*** ')
            st.markdown('噪音大小')
            st.markdown("""
                `美的产品噪音较小，但与其他产品差别不大`
            """)
            st.markdown('''
                        **[美的](https://dyf-2316.github.io/HYF_LDA_results/tag/美的（Midea）噪音大小_comment.html)**\n
                        **[海尔](https://dyf-2316.github.io/HYF_LDA_results/tag/海尔（Haier）噪音大小_comment.html)**\n
                        **[史密斯](https://dyf-2316.github.io/HYF_LDA_results/tag/史密斯（A.O.S噪音大小_comment.html)**\n

                    ''')

        elif feature == '出水速度':
            st.markdown('*** ')
            st.markdown('出水速度')
            st.markdown("""
                `美的产品出水速度快，温度稳定，但与其他产品差别不大`
            """)
            st.markdown('''
                        **[美的](https://dyf-2316.github.io/HYF_LDA_results/tag/美的（Midea）出水速度_comment.html)**\n
                        **[海尔](https://dyf-2316.github.io/HYF_LDA_results/tag/海尔（Haier）出水速度_comment.html)**\n
                        **[史密斯](https://dyf-2316.github.io/HYF_LDA_results/tag/史密斯（A.O.S出水速度_comment.html)**\n

                    ''')



def show_time_seq():
    st.markdown('***')
    st.subheader('评论数量及评分随时间变化的结果')
    st.markdown('>揭示一段时间内产品评价量和平均评分变化，并可视化处理')
    if st.button('show code detail',key='time_seq'):
        st.code('''import pandas as pd
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
infile="compressed_comment.csv"
data=pd.read_csv(infile,encoding='utf-8',header=None)
data.head()
df=pd.DataFrame(data)
df[3]=pd.to_datetime(df[3],errors='coerce')
df.set_index(df[3],inplace=True)
#数量
num_of_comment=pd.DataFrame(df['2019-01':'2020-07'][2].astype(int).resample('M').size())
num_of_comment.index.name='时间'
num_of_comment.name='num_of_comment'
#均值
avg_of_score=pd.DataFrame(df['2019-01':'2020-07'][2].astype(int).resample('M').mean())
avg_of_score.index.name='时间'
avg_of_score.name='avg_of_score'
#合并
res=pd.merge(pd.DataFrame(num_of_comment),pd.DataFrame(avg_of_score),left_index=True,right_index=True)
res=pd.DataFrame(res)
res.rename(columns={'2_x':'num_of_comment','2_y':'avg_of_score'},inplace=True)
print(res)
#ln&e
num_of_comment_ln=pd.DataFrame(df['2019-01':'2020-07'][2].astype(float).resample('M').size().apply(np.log1p))
num_of_comment_ln.index.name='时间'
res2=pd.merge(pd.DataFrame(num_of_comment_ln),pd.DataFrame(avg_of_score),left_index=True,right_index=True)
res2=pd.DataFrame(res2)
res2.rename(columns={'2_x':'num_of_comment_ln','2_y':'avg_of_score'},inplace=True)
print(res2)

#plt
x=res.index
x=pd.to_datetime(x,errors='coerce')
y1=res['num_of_comment']
y2=res2['num_of_comment_ln']
y3=res2['avg_of_score']
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
#plt.plot(x, y1, color='teal', linewidth=1.0, linestyle='-', label='predict')#原评论
plt.plot(x, y2, color='navy', linewidth=2.0, linestyle='-', label='predict')#对数后的评论量
plt.plot(x, y3, color='orange', linewidth=1.0, linestyle='--', label='predict')#平均分
plt.xlabel('TIME')
plt.ylabel('SCORE/AMOUNT(log)')
plt.legend(['Amount','Score'],loc='upper right')
plt.title(r'$Amount(log)/Score---Time$',fontsize=25,color='teal')
plt.show()


    ''')
    show_image(r'score_amount_timeSeq.png', '时间序列结果')


def show_netanalysis():
    st.markdown('***')
    st.subheader('评论数量及评分随时间变化的结果')
    st.markdown('''
    >1.使用jieba分词，删除停用词

>2.Collections统计词频，筛选高频词

>3.构建语义关联矩阵

>4.对生成的语义关联矩阵进行权重的归一化

>5.删除权重较小的边

>6.用networkx生成语义网络图

>7.将语义网络图保存为png格式
''')
    if st.button('show code detail',key='netanalysis'):
        st.code('''
        import re  # 正则表达式库
import jieba  # 分词
import collections  # 词频统计库
import numpy as np
import pandas as pd
import networkx as nx  # 复杂网络分析库
import matplotlib.pyplot as plt


# print(dir(nx.layout))
def networkx_analysis(dir='美的（Midea）JSQ22-L1(Y)_comment_正面.csv'):
    """
    输入待分析的csv文件路径
    输出networkx网络分析结果
    """
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
    file = open('../../resources/stopwords.txt', 'r', encoding='utf-8').readlines()  # 自定义去除词库
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


        ''')
    show_image(r'语义网络.png', '语义网络')
    show_image(r'rost语义网络.png', 'rost语义网络')



show_crawler()
show_LDA_visualization()
show_time_seq()
show_netanalysis()
