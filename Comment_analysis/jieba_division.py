import re # 正则表达式库
import jieba   #分词
import jieba.posseg as pseg
import pandas as pd

def division(dir):
    '''

    :param dir:文件路径
    :return:
    '''
    pattern = re.compile(r'([^/\\:]+)\.csv')
    data = pattern.search(dir)
    title = data[0]
    title = title.strip('.csv')

    fn = pd.read_csv(dir, encoding='utf-8', engine='python')  # 打开文件
    string_data = fn['comment'].tolist()  # 读出评论文件
    stop_words = []
    word_pos_list = ['n', 'vn', 'a', 'ad']
    jieba.enable_paddle()
    file = open('stopwords.txt', 'r', encoding='utf-8').readlines()  # 自定义去除词库
    for each_line in file:
        each_line = each_line.strip('\n')
        stop_words.append(each_line)

    # pattern = re.compile(u'\t|。|，|：|；|！|）|（|？|、|“|”')  # 定义正则表达式匹配模式
    with open(title + '_分词.txt', 'w', encoding='utf-8') as fp:
        for i in string_data:
            # processed_str = re.sub(pattern, '', string_data[i])
            seg_list_exact = pseg.cut(i, use_paddle=True)  # 精确模式分词
            object_list = []
            for word, flag in seg_list_exact: # 循环读出每个分词

                if word not in stop_words and flag in word_pos_list: # 如果不在去除词库中
                    object_list.append(word)
            if not len(object_list) == 0:
                print(''.join(object_list))
                fp.write(''.join(object_list)+'\n')



division("../resources/data/meidi_yearly_comment/spider_meidi_comments2019.csv")