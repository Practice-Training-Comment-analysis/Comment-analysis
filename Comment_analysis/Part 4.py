import pandas as pd
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
