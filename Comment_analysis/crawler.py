"""
open ----------- encoding='utf-8'
beautiful soup ------ features='lxml'
xpath 解析结果记得加[0]

实现细节
品牌 商品名称 评论
获取productID以实现不同页面爬取https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100011742150&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1
score=0/5
page要变
"""

# %%

from urllib import request
import pymysql
import requests
from lxml import etree
import time
import re
import json
import csv
import logging

import inspect
from resources.configs import config

"""
:param
        'page': 2i+1,
        's': 50i+1,
"""

logger = logging.getLogger('crawler logger')


def log(e=None, message=None):
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
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("../resources/logs/log.txt", encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(handler)
        logger.addHandler(console)
    if message is not None:
        logger.info(message)
    # logger.debug("Do something")
    # "Automatically log the current function details."
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    if e is not None:
        # Dump the message + the name of this function to the log.
        logger.error("%s:  %s  in  %s :  %i" % (
            e.__str__(),
            func.co_name,
            func.co_filename,
            func.co_firstlineno
        ))
        print(e.__str__(),
              func.co_name,
              func.co_filename,
              func.co_firstlineno)
        logger.warning("Something maybe fail.")
    # logger.info("Finish")


def get_product_id():
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        'cookie': 'unpl=V2_ZzNtbRBSF0JxWxFdfR1eUWIFFw4RV0ZAJQtOVC4fXlBlChUNclRCFnQUR1NnGVUUZgsZXkBcQBNFCEdkeR1ZAmYBEV1yZ3MWdThHZHsdVAZiBhBYQ1BKFXQLR1R4HFwBZQsWbXJQQxxFOA0BJ3caRCUzEl1FU0YXfQpPVUsYbAdmAhFZQ1dBHXE4DTp6VFwBbwAXWEBSQhJ8CEdXehlfAGcHEFVGZ0MUdAlHVH8aXQRXAA%3d%3d; shshshfpa=ce0402f7-cd1b-b741-3882-419b5f2a8d98-1592443409; __jdu=15924434068102012415395; shshshfpb=nUtzw%2FiFPlAlC2mX1q6G7kQ%3D%3D; areaId=3; PCSYCityID=CN_120000_120100_120112; user-key=378b9912-73ca-4776-a9bf-cef8f17f7b10; cn=0; rkv=V0800; ipLoc-djd=3-51047-55748-0; __jdc=122270672; shshshfp=4fc33d688da9419110a1f4af5bdf991c; 3AB9D23F7A4B3C9B=6LY7NRXYJ5ZJJHRFEFRYRQZQQ6GTYX5G7H4CWY3TFERY673ERRPOCDPMWNIWBDA5D3U7NJTJNQA2F36SIKNUT2N4RQ; qrsc=3; __jdv=122270672|direct|-|none|-|1593780569513; __jda=122270672.15924434068102012415395.1592443406.1593658621.1593780570.5'
    }
    root_url = 'https://search.jd.com/Search?'
    ids = []
    # 用beautiful soup更方便
    pattern_id = re.compile('<li\sdata-sku="(\d*)".*?>', re.S)

    for i in range(20):
        print("开始爬取第", i)
        params = {
            'keyword': '美的热水器',
            'qrst': '1',
            'suggest': '1.his.0.0',
            'wq': '美的热水器',
            'stock': '1',
            'ev': 'exbrand_美的（Midea） ^',
            'cid3': '13691',
            'page': 2 * i + 1,
            's': 50 * i + 1,
            'click': '0'
        }

        try:
            response = requests.get(root_url, headers=headers, params=params)
            response.encoding = 'utf-8'
            html = response.text
            id = pattern_id.findall(html)
            ids.append(id)
            print("结束爬取", i)
            time.sleep(2)
        except Exception as e:
            log(e)
            with open('../resources/data/test_product_id.csv', 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(ids)

    with open('../resources/data/test_product_id.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(ids)


def get_page(url, headers):
    try:

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            response.encoding = 'utf-8'
            if response.text == '':
                time.sleep(300)
                return get_page(url, headers)
            return response.text
        return None
    except requests.RequestException:
        return None


def get_basic_product_info(id):
    """
    get price brand model  goodCount, generalCount, poorCount
    :param id:
    :return:
    """
    try:

        # get page html
        page_url = 'https://item.jd.com/' + id + '.html'
        page_html = get_page(page_url, config.HEADERS)
        if page_html == '':
            raise Exception('get_basic_product_info : html contains nothing')

        # 通过价格所在json 的url来进行获取
        price_url = "https://p.3.cn/prices/mgets?skuIds=J_" + id
        url_session = requests.Session()
        price_req = url_session.get(price_url).text
        price = float(re.findall(r'"p":"(.*?)"', price_req)[0])

        # 创建html树，进行xpath索引brand,re索引model
        selector = etree.HTML(page_html)
        brand = selector.xpath('//*[@id="detail"]/div[2]/div[1]/div[1]/ul[1]/li/@title')[0]
        # models = re.findall('<ul class="parameter2.*?货号：(.*?)</li>|<ul class="parameter2.*?型号：(.*?)</li>|<dl class="clearfix".*?<dt>型号.*?<dd>(.*?)</dd>', page_html, re.S)
        models = re.findall('<dl class="clearfix".*?<dt>型号.*?<dd>(.*?)</dd>', page_html, re.S)
        if models is not None:
            model = models[0]
        else:
            model = None

        # 获取评论数量相关信息
        url1 = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment198&productId=' + str(
            id) + '&score=0&sortType=5&page=1&pageSize=10&isShadowSku=0&fold=1'
        html1 = request.urlopen(url1).read().decode('gbk', 'ignore')
        goodCount, generalCount, poorCount = None, None, None
        if html1 != '':
            # raise Exception('get_basic_product_info : line 165 : html1 contains nothing')
            info_json1 = json.loads(html1[21:-2], encoding='utf-8')
            if info_json1 != {}:
                goodCount = dict(info_json1)['productCommentSummary']['goodCount']
                generalCount = dict(info_json1)['productCommentSummary']['generalCount']
                poorCount = dict(info_json1)['productCommentSummary']['poorCount']
                print(goodCount, generalCount, poorCount)
        else:

            time.sleep(600)
            raise Exception('get_basic_product_info :  html1 contains nothing')
        return brand, model, price, goodCount, generalCount, poorCount
    except Exception as e:
        log(e)
        return None, None, None, None, None, None


def get_comments_and_to_file(product_id, page_count, comment_score=0):
    """
    get comments from json

    :param page_count:
    :param score:
    :parameter url: json url
    :param product_id:

    :return:
    """
    # key_words = ['外形外观：',
    #              '恒温效果：',
    #              '噪音大小：',
    #              '出水速度：',
    #              '安装服务：']
    try:

        for p in range(page_count):
            print(p)
            score = []
            comment_time = []
            comments = []

            url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment198&productId=' + str(
                product_id) + '&score=' + str(comment_score) + '&sortType=5&page=' + str(
                p + 1) + '&pageSize=10&isShadowSku=0&fold=1'
            html = request.urlopen(url).read().decode('gbk', 'ignore')
            if html == '':
                raise Exception('html contains nothing')
            # print(html[21:-2])
            info_json = json.loads(html[21:-2], encoding='utf-8')
            for i in range(10):
                if info_json['comments'][i] is None:
                    break
                comm_info = dict(info_json['comments'][i])

                if comm_info['content'] is None or comm_info['creationTime'] is None or comm_info['score'] is None:
                    break
                else:
                    comments.append(comm_info['content'])
                    comment_time.append(comm_info['creationTime'])
                    score.append(comm_info['score'])
            time.sleep(3)
            if not (score == [] or comments == [] or comment_time == []):
                id_list = [product_id] * len(comments)
                comments_to_file(id_list, comments, comment_time, score)

        # return comments, comment_time, score
    except Exception as e:

        log(e)
        print('get comments and to file failed' + e.__str__())
        # return comments, comment_time, score


def info_to_file():
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

        # write information into table product info
        with open('../resources/data/test_product_id.csv', 'r', encoding='utf-8') as csvFile:
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





    except Exception as e:
        log(e)
        print('info_to_file failed')

    finally:
        connection.close()


def comments_to_file(product_id, comments, comment_time, score):
    """
    store comment related information into database
    :param product_id: list
    :param comments: list
    :param comment_time: list
    :param score: list
    :return:
    """
    # print(product_id, comments, comment_time, score)

    try:
        with connection.cursor() as cursor:
            # 写数据库
            sql = "insert into spider.comments(product_id, score, comment_time, comment) values(%s,%s,%s,%s)"
            for i in range(len(comments)):
                cursor.execute(sql, (product_id[i], score[i], comment_time[i], comments[i]))
            connection.commit()

    except Exception as e:
        log(e)
        print('comments_to_file failed')


info_to_file()
