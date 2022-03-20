
##  套件.
import feature
import library
import data

##  宣告暫存物件.
cache = feature.cache(storage='resource/preprocess')

##  載入所有的表格資料.
table = data.table(source='kaggle')

##  針對 article 表進行前處理.
##  補缺失值, 對 target 進行編碼並新增一個欄位存放.
##  針對其他類別變數進行編碼, 並保留 [0, 1, 2] 分別給 [<padding>, <start previous>, <start future>] 使用.
cache.article = table.article.copy()
cache.article["detail_desc"] = cache.article["detail_desc"].fillna("missing value")
cache.article['article_code'] = feature.category.encode(cache.article['article_id']) + 3
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
    
    cache.article[l] = feature.category.encode(cache.article[l]) + 3
    pass

default = [
    ["<padding>"]+[0 for _ in range(25)], 
    ["<start previous>"]+[1 for _ in range(25)], 
    ["<start future>"]+[2 for _ in range(25)]
]
default = library.pandas.DataFrame(default, columns=cache.article.columns)
cache.article = library.pandas.concat([default, cache.article])
cache.save(what=cache.article, file='article.csv', format='csv')

##  針對 transaction 表進行前處理, 以用戶當作 row 來建構對應的標記與特徵序列.
table.transaction = table.transaction
table.transaction['sales_channel_id'] = feature.category.encode(table.transaction['sales_channel_id']) + 3
table.transaction['price'] = 1 + (table.transaction['price'] / table.transaction['price'].max())
selection = ['article_code'] + [
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
for s in selection:

    c = library.pandas.merge(table.transaction, cache.article[["article_id", s]], on="article_id", how='inner').copy()
    c[s] = c[s].astype(str)
    row[s] = c[['customer_id', 't_dat', s]].groupby(['customer_id', 't_dat'])[s].apply(" ".join).reset_index()
    row[s] = row[s][['customer_id', 't_dat', s]].groupby(['customer_id'])[s].apply(" ".join).reset_index()
    pass

for s in ['price', "sales_channel_id"]:

    table.transaction[s] = table.transaction[s].astype(str)    
    c = table.transaction[['customer_id', 't_dat', s]]
    c = c.groupby(['customer_id', 't_dat'])[s].apply(" ".join).reset_index()
    c = c[['customer_id', 't_dat', s]].groupby(['customer_id'])[s].apply(" ".join).reset_index()
    row[s] = c
    pass

row = [row[k] for k in row.keys()]
cache.transaction = library.functools.reduce(
    lambda x,y: library.pandas.merge(left=x, right=y, on='customer_id', how='inner'), 
    row
)
cache.save(what=cache.transaction, file='transaction.csv', format='csv')

##  針對 customer 表進行前處理.
cache.customer = table.customer.copy()
cache.customer['FN'] = cache.customer['FN'].fillna(0.0)
cache.customer['Active'] = cache.customer['Active'].fillna(0.0)
cache.customer['club_member_status'] = cache.customer['club_member_status'].fillna("MISS")
cache.customer['fashion_news_frequency'] = cache.customer['fashion_news_frequency'].fillna("MISS")
cache.customer['age'] = cache.customer['age'].fillna(36.0)
cache.customer['club_member_status'] = feature.category.encode(cache.customer['club_member_status']) + 0
cache.customer['fashion_news_frequency'] = feature.category.encode(cache.customer['fashion_news_frequency']) + 0
cache.customer['age'] = cache.customer['age'] / 100
cache.customer['postal_code'] = feature.category.encode(cache.customer['postal_code']) + 0
cache.save(what=cache.customer, file='customer.csv', format='csv')

##  整合特徵.
cache.f1 = library.pandas.merge(left=cache.customer, right=cache.transaction, on='customer_id', how='inner')
cache.f1['seq_len'] = cache.f1['article_code'].apply(lambda x: len(x.split()))
cache.save(cache.f1, 'f1.csv', 'csv')

##  整合類別種類.
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
