
import os
import pandas
import tqdm
import functools
from sklearn import preprocessing

source = './resource/kaggle/sample/csv'
table = {
    'article':pandas.read_csv(os.path.join(source, 'articles.csv'), dtype=str),
    'customer':pandas.read_csv(os.path.join(source, 'customers.csv'), dtype=str),
    'transaction':pandas.read_csv(os.path.join(source, 'transactions_train.csv'), dtype=str),
    'submission':pandas.read_csv(os.path.join(source, 'sample_submission.csv'), dtype=str)
}

table['article']['detail_desc'].fillna('miss_detail_desc', inplace=True)
table['customer']['FN'].fillna("0.0", inplace=True)
table['customer']['Active'].fillna("0.0", inplace=True)
table['customer']['fashion_news_frequency'].fillna("NONE", inplace=True)
table['customer']['age'].fillna("-1", inplace=True)
table['customer']['postal_code'].fillna('miss_postal_code', inplace=True)

table['article']["article_label"] = table['article']['article_id']
for key in table['article']:

    if(key=='article_id'): continue

    skip = 10
    engine = preprocessing.LabelEncoder()
    engine.fit(table['article'][key])
    value = engine.transform(table['article'][key])
    table['article'][key] = value + skip
    continue

table['customer']['FN'] = table['customer']['FN'].astype(float)
table['customer']['Active'] = table['customer']['Active'].astype(float)
table['customer']['age'] = table['customer']['age'].astype(float) / 100
for key in table['customer']:

    if(key in ['customer_id', "FN", "Active", 'age']): continue

    skip = 10
    engine = preprocessing.LabelEncoder()
    engine.fit(table['customer'][key])
    value = engine.transform(table['customer'][key])
    table['customer'][key] = value + skip
    continue

table['transaction']['t_dat'] = pandas.to_datetime(table['transaction']['t_dat'])
value = table['transaction']['price'].astype(float).copy()
value = value / value.max()
table['transaction']['price'] = value
for key in table['transaction']:
    
    if(key in ['t_dat', "customer_id", "article_id", 'price']): continue

    skip = 10
    engine = preprocessing.LabelEncoder()
    engine.fit(table['transaction'][key])
    value = engine.transform(table['transaction'][key])
    table['transaction'][key] = value + skip
    continue

paste = lambda x: " ".join(x)

sequence = []
table['transaction']['price'] = table['transaction']['price'].astype(str)
table['transaction']['sales_channel_id'] = table['transaction']['sales_channel_id'].astype(str)
sequence += [table['transaction'].groupby(['customer_id', 't_dat'])['price'].apply(paste).reset_index().drop(columns='t_dat')]
sequence += [table['transaction'].groupby(['customer_id', 't_dat'])['sales_channel_id'].apply(paste).reset_index().drop(columns='t_dat')]

for key in tqdm.tqdm(table['article'].keys()):
    
    if(key=='article_id'): continue

    table['article'][key] = table['article'][key].astype(str)
    selection = table['transaction'][['t_dat', "customer_id", "article_id"]].copy()
    selection = pandas.merge(selection, table['article'][["article_id", key]], on="article_id", how='inner')
    sequence += [selection.groupby(['customer_id', 't_dat'])[key].apply(paste).reset_index().drop(columns='t_dat')]
    continue

sequence = {k:v for k,v in enumerate(sequence)}

merge = lambda x,y: pandas.merge(left=x, right=y, on='customer_id', how='inner')
sequence = functools.reduce(merge, sequence.values())

