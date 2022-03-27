
##  載入套件.
import feature
import library
import data

##  載入資料.
table = data.table(source='kaggle', mode='sample')
pass

##  初步清理.
table.article["detail_desc"] = table.article["detail_desc"].fillna("missing value")
table.customer['FN'] = table.customer['FN'].fillna(0.0)
table.customer['Active'] = table.customer['Active'].fillna(0.0)
table.customer['club_member_status'] = table.customer['club_member_status'].fillna("MISS")
table.customer['fashion_news_frequency'] = table.customer['fashion_news_frequency'].fillna("MISS")
table.customer['age'] = table.customer['age'].fillna(36.0)
pass

##  
cache = feature.cache(storage='resource/preprocess')
pass

##  針對 article 表進行前處理.
table.article['article_code'] = table.article['article_id']
reservation = library.pandas.DataFrame(columns=table.article.columns)
reservation.loc[0] = 0
reservation.loc[1] = 1
reservation.loc[2] = 2
reservation['article_id'] = ['<padding>', "<start>", "<end>"]
loop = [
    'article_code', 'product_code', 'prod_name', 'product_type_no',
    'product_type_name', 'product_group_name', 'graphical_appearance_no',
    'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name',
    'perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name',
    'index_group_no', 'index_group_name', 'section_no', 'section_name',
    'garment_group_no', 'garment_group_name', 'detail_desc'
]
for l in loop: 
    
    table.article[l] = feature.label.encode(table.article[l]) + len(reservation)
    pass

cache.article = library.pandas.concat([reservation, table.article]).reset_index(drop=True)
cache.save(what=cache.article, file='article.csv', format='csv')
pass

##  針對 transaction 表進行前處理, 以用戶當作 row 來建構對應的標記與特徵序列.
table.transaction['date_code'] = feature.label.encode(table.transaction['t_dat']) + len(reservation)
table.transaction['sales_channel_id'] = feature.label.encode(table.transaction['sales_channel_id']) + len(reservation)
table.transaction['price'] = 1 + (table.transaction['price'] / table.transaction['price'].max())
pass

sequence = dict()
pass

v = 'price'
c = table.transaction.copy().astype(str)
sequence[v] = feature.sequence.flatten(table=c, key='customer_id', group=['customer_id', 't_dat'], variable=v)
pass

v = 'sales_channel_id'
c = table.transaction.copy().astype(str)
sequence[v] = feature.sequence.flatten(table=c, key='customer_id', group=['customer_id', 't_dat'], variable=v)
pass

v = 'date_code'
c = table.transaction.copy().astype(str)
sequence[v] = feature.sequence.flatten(table=c, key='customer_id', group=['customer_id', 't_dat'], variable=v)
pass

v = 'article_code'
c = library.pandas.merge(
    table.transaction, 
    cache.article[["article_id", v]], 
    on="article_id", how='inner'
).copy().astype(str)
sequence[v] = feature.sequence.flatten(table=c, key='customer_id', group=['customer_id', 't_dat'], variable=v)
pass

sequence = sequence.values()
sequence = library.functools.reduce(
    lambda x,y: library.pandas.merge(left=x, right=y, on='customer_id', how='inner'), 
    sequence
)
cache.sequence = sequence
cache.save(what=cache.sequence, file='sequence.csv', format='csv')
pass

##  針對 customer 表進行前處理.
table.customer['club_member_status'] = feature.label.encode(table.customer['club_member_status']) + 0
table.customer['fashion_news_frequency'] = feature.label.encode(table.customer['fashion_news_frequency']) + 0
table.customer['age'] = table.customer['age'] / 100
table.customer['postal_code'] = feature.label.encode(table.customer['postal_code']) + 0
cache.customer = table.customer
cache.save(what=cache.customer, file='customer.csv', format='csv')
pass

##  整合特徵.
cache.f1 = library.pandas.merge(left=cache.customer, right=cache.sequence, on='customer_id', how='inner')
cache.f1['trans_length'] = cache.f1['date_code'].apply(lambda x: len(x.split()))
cache.save(cache.f1, 'f1.csv', 'csv')
pass
