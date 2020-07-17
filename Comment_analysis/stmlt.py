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

    st.markdown('''**数据采集和抽取**
    
    
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

    display_type = st.selectbox('请选择', ('美的品牌评论情感分类LDA结果', '美的品牌评论随时间变化LDA结果', '美的品牌与其他品牌热水器在各方面评论对比LDA结果'))
    if display_type == '美的品牌评论情感分类LDA结果':

        show_image('meidi_pos.png', 'meidi_pos')
        show_image('meidi_neg.png', 'meidi_neg')

    elif display_type == '美的品牌评论随时间变化LDA结果':
        show_image('meidi2018.jpg', 'meidi2018')
        show_image('meidi2019.jpg', 'meidi2019')
        show_image('meidi2020.jpg', 'meidi2020')


    elif display_type == '美的品牌与其他品牌热水器在各方面评论对比LDA结果':
        feature = st.radio('请选择一方面进行对比', ['安装服务', '外形外观'])
        if feature == '安装服务':
            show_image('meidi_service.png', 'meidi_service')
            show_image('haier_service.png', 'haier_service')
        if feature == '外形外观':
            show_image('meidi_appearance.png', 'meidi_appearance')
            show_image('haier_appearance.png', 'haier_appearance')
