import pandas as pd
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
res.rename(columns={0:'num_of_comment',1:'avg_of_score'},inplace=True)
print(res)

