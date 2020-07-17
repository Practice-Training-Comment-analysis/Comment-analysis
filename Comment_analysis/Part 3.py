
import pandas as pd
import re
import jieba
import collections

#获取词汇列表
num=50
fi1=open("../resources/data/compressed_comment.csv",'r',encoding='utf-8')#jd满分评价.txt
strdata=fi1.read()
fi1.close()

pattern=re.compile(u'\t|\.|,|:|;|!|\)|\(|\?|"')
strdata=re.sub(pattern,'',strdata)
strdata=re.sub("[\s+\.\!V,$%^*(+\"\"]+|[-+!。！？，,.?、~@#$%......&*();`:]+","",strdata)

seglist=jieba.cut(strdata,cut_all=False)

obj=[]
sw=list(open("../resources/stopwords.txt",'r',encoding='utf-8').read())
sw.append("\n")
#print(sw)

for word in seglist:
    if word not in sw:
        obj.append(word)
# print("obj个数",len(obj))

wc=collections.Counter(obj)
wct=wc.most_common(num)
word=pd.DataFrame(wct,columns=["关键词","次数"])
target_word=list(word['关键词'])
#筛选

file="../resources/data/compressed_comment.csv"
data=pd.read_csv(file,encoding="utf-8")
data.drop_duplicates(['comment'],inplace=True)

good_value_list=[]
bad_value_list=[]
def word_process(no):
    goodlist = []
    badlist = []
    testlist = []
    for idx, line in data.iterrows():
        all_list = re.findall(r"^{}(.+?)\n".format(target_word[no]), line['comment'] + "\n", re.M)
        if not len(all_list) == 0:
            testlist.append((line['score'], all_list[0]))
            if line['score'] == 1 or line['score'] == 2:
                badlist.append((line['score'], all_list[0]))
            elif line['score'] == 5 or line['score'] == 4:
                goodlist.append((line['score'], all_list[0]))
            else:
                pass
    goodvalue = len(goodlist)
    badvalue = len(badlist)
    good_value_list.append(goodvalue)
    bad_value_list.append(badvalue)
    for i in range(num):
        word_process(i)

    word['好值']=good_value_list
    word['坏值']=bad_value_list
    #取结果
    good_seq=word.sort_values(by='好值' , ascending=False)
    bad_seq=word.sort_values(by='坏值' , ascending=False)
    res_good=list(good_seq['关键词'])[:5]
    res_bad=list(bad_seq['关键词'])[:5]
    return res_good,res_bad




