
##  套件.
import feature
import library
import data

##  宣告暫存物件.
cache = feature.cache(storage='resource/preprocess(sample)')

##  載入所有的表格資料.
table = data.table(source='kaggle', sample=True)

##  針對 article 表進行前處理.
table.article["detail_desc"] = table.article["detail_desc"].fillna("missing value")
reservation = ['<padding>', "<history>", "<future>"]
table.article['article_code'] = feature.category.encode(table.article['article_id']) + len(reservation)
loop = [
    'product_code', 'prod_name', 'product_type_no',
    'product_type_name', 'product_group_name', 'graphical_appearance_no',
    'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name',
    'perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name',
    'index_group_no', 'index_group_name', 'section_no', 'section_name',
    'garment_group_no', 'garment_group_name', 'detail_desc'
]
for l in loop: 
    
    table.article[l] = feature.category.encode(table.article[l]) + len(reservation)
    pass

default = library.pandas.DataFrame(columns=table.article.columns)
default.loc[0] = 0
default.loc[1] = 1
default.loc[2] = 2
default['article_id'] = reservation
cache.article = library.pandas.concat([default, table.article]).reset_index(drop=True)
cache.save(what=cache.article, file='article.csv', format='csv')

##  針對 transaction 表進行前處理, 以用戶當作 row 來建構對應的標記與特徵序列.
<<<<<<< HEAD
table.transaction = table.transaction
table.transaction['sales_channel_id'] = feature.category.encode(table.transaction['sales_channel_id']) + 3
=======
table.transaction['sales_channel_id'] = feature.category.encode(table.transaction['sales_channel_id']) + len(reservation)
>>>>>>> 499600923f8d04cae668a1057f89a269292cb673
table.transaction['price'] = 1 + (table.transaction['price'] / table.transaction['price'].max())
loop = ['article_code'] + [
    'product_code', 'prod_name', 'product_type_no',
    'product_type_name', 'product_group_name', 'graphical_appearance_no',
    'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name',
    'perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name',
    'index_group_no', 'index_group_name', 'section_no', 'section_name',
    'garment_group_no', 'garment_group_name', 'detail_desc'
]
row = {}
for l in loop:

    c = library.pandas.merge(table.transaction, cache.article[["article_id", l]], on="article_id", how='inner').copy()
    c[l] = c[l].astype(str)
    row[l] = c[['customer_id', 't_dat', l]].groupby(['customer_id', 't_dat'])[l].apply(" ".join).reset_index()
    row[l] = row[l][['customer_id', 't_dat', l]].groupby(['customer_id'])[l].apply(" ".join).reset_index()
    pass

for l in ['price', "sales_channel_id"]:

    table.transaction[l] = table.transaction[l].astype(str)    
    c = table.transaction[['customer_id', 't_dat', l]]
    c = c.groupby(['customer_id', 't_dat'])[l].apply(" ".join).reset_index()
    c = c[['customer_id', 't_dat', l]].groupby(['customer_id'])[l].apply(" ".join).reset_index()
    row[l] = c
    pass

row = [row[k] for k in row.keys()]
cache.transaction = library.functools.reduce(
    lambda x,y: library.pandas.merge(left=x, right=y, on='customer_id', how='inner'), 
    row
)
cache.save(what=cache.transaction, file='transaction.csv', format='csv')

##  針對 customer 表進行前處理.
table.customer['FN'] = table.customer['FN'].fillna(0.0)
table.customer['Active'] = table.customer['Active'].fillna(0.0)
table.customer['club_member_status'] = table.customer['club_member_status'].fillna("MISS")
table.customer['fashion_news_frequency'] = table.customer['fashion_news_frequency'].fillna("MISS")
table.customer['age'] = table.customer['age'].fillna(36.0)
table.customer['club_member_status'] = feature.category.encode(table.customer['club_member_status']) + 0
table.customer['fashion_news_frequency'] = feature.category.encode(table.customer['fashion_news_frequency']) + 0
table.customer['age'] = table.customer['age'] / 100
table.customer['postal_code'] = feature.category.encode(table.customer['postal_code']) + 0
cache.customer = table.customer
cache.save(what=cache.customer, file='customer.csv', format='csv')

##  整合特徵.
cache.f1 = library.pandas.merge(left=cache.customer, right=cache.transaction, on='customer_id', how='inner')
cache.f1['seq_len'] = cache.f1['article_code'].apply(lambda x: len(x.split()))
cache.save(cache.f1, 'f1.csv', 'csv')

<<<<<<< HEAD
##  整合類別種類.
=======
# ##  整合類別種類.
>>>>>>> 499600923f8d04cae668a1057f89a269292cb673
# cache.embedding = dict()
# loop = ['postal_code'] + ['article_code'] +[
#     'product_code', 'prod_name', 
#     'product_type_no', 'product_type_name',
#     'product_group_name', 'graphical_appearance_no',
#     'graphical_appearance_name', 'colour_group_code', 
#     'colour_group_name', 'perceived_colour_value_id', 
#     'perceived_colour_value_name', 'perceived_colour_master_id', 
#     'perceived_colour_master_name', 'department_no', 
#     'department_name', 'index_code', 
#     'index_name', 'index_group_no', 
#     'index_group_name', 'section_no', 
#     'section_name', 'garment_group_no', 
#     'garment_group_name', 'detail_desc'
# ] + ['sales_channel_id']
# for l in loop:

#     if(l=="postal_code"):

#         v = cache.customer[l].astype(int)
#         pass

#     elif(l=="sales_channel_id"):

#         v = [0,1,2] + [int(j) for i in cache.f1['sales_channel_id'] for j in i.split()]
#         v = library.pandas.Series(v)
#         pass

#     else:

#         v = cache.article[l].astype(int)
#         pass
    
#     s = (
#         min(v),
#         max(v), 
#         v.nunique()
#     )
#     cache.embedding[l] = ["(min:{},max:{},unique:{})".format(s[0], s[1], s[2])]
#     pass

# cache.embedding = library.pandas.DataFrame(cache.embedding)
# cache.save(cache.embedding, 'embedding.csv', 'csv')
