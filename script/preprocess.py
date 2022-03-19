
##  套件.
import feature
import library
import data

##  宣告暫存物件.
cache = data.cache(storage='resource/preprocess')

##  載入所有的表格資料.
table = data.table(source='kaggle')

##  針對 article 表進行前處理.
##  補缺失值, 對 target 進行編碼並新增一個欄位存放.
##  針對其他類別變數進行編碼, 並保留 [0, 1, 2] 分別給 [<padding>, <start previous>, <start future>] 使用.
cache.article = table.article.copy()
cache.article["detail_desc"] = cache.article["detail_desc"].fillna("missing value")
cache.article['article_code'] = feature.category.encode(cache.article['article_id'], 3)
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
    
    cache.article[l] = feature.category.encode(cache.article[l], 3)
    pass

default = [["<padding>"]+[0 for _ in range(25)], ["<start previous>"]+[1 for _ in range(25)], ["<start future>"]+[2 for _ in range(25)]]
default = library.pandas.DataFrame(default, columns=cache.article.columns)
cache.article = library.pandas.concat([default, cache.article])
cache.save(what=cache.article, file='article.csv', format='csv')
cache.save(what=library.pandas.DataFrame(cache.article.apply(lambda x: x.nunique())).transpose(), file='article embedding information.csv', format='csv')

##  針對 transaction 表進行前處理, 以用戶當作 row 來建構對應的標記與特徵序列.
table.transaction = table.transaction
table.transaction['sales_channel_id'] = feature.category.encode(table.transaction['sales_channel_id'], 3)
table.transaction['price'] = 1 + (table.transaction['price'] / table.transaction['price'].max())
selection = [
    'product_code', 'prod_name', 'product_type_no',
    'product_type_name', 'product_group_name', 'graphical_appearance_no',
    'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name',
    'perceived_colour_master_id', 'perceived_colour_master_name',
    'department_no', 'department_name', 'index_code', 'index_name',
    'index_group_no', 'index_group_name', 'section_no', 'section_name',
    'garment_group_no', 'garment_group_name', 'detail_desc',
    'article_code'
]
row = {}
for s in selection:

    c = library.pandas.merge(table.transaction, cache.article[["article_id", s]], on="article_id", how='inner').copy()
    c[s] = c[s].astype(str)
    row[s] = c[['customer_id', 't_dat', s]].groupby(['customer_id', 't_dat'])[s].apply(" ".join).reset_index()
    row[s] = row[s][['customer_id', 't_dat', s]].groupby(['customer_id'])[s].apply(" ".join).reset_index()
    pass

table.transaction['price'] = table.transaction['price'].astype(str)
table.transaction['sales_channel_id'] = table.transaction['sales_channel_id'].astype(str)
row['price'] = table.transaction[['customer_id', 't_dat', 'price']].groupby(['customer_id', 't_dat'])['price'].apply(" ".join).reset_index()
row['price'] = row['price'][['customer_id', 't_dat', 'price']].groupby(['customer_id'])['price'].apply(" ".join).reset_index()
row['sales_channel_id'] = table.transaction[['customer_id', 't_dat', 'sales_channel_id']].groupby(['customer_id', 't_dat'])['sales_channel_id'].apply(" ".join).reset_index()
row['sales_channel_id'] = row['sales_channel_id'][['customer_id', 't_dat', 'sales_channel_id']].groupby(['customer_id'])['sales_channel_id'].apply(" ".join).reset_index()
row = [row[k] for k in row.keys()]
cache.transaction = library.functools.reduce(lambda x,y: library.pandas.merge(left=x, right=y, on='customer_id', how='inner'), row)
cache.save(what=cache.transaction, file='transaction.csv', format='csv')


# cache.transaction = library.pandas.merge(table.transaction, cache.article[selection], on="article_id", how='inner').copy()

# for k in cache.transaction.keys():

#     cache.transaction[k] = cache.transaction[k].astype(str)
#     pass

# cache.transaction['article_code'] = cache.transaction['article_code'].astype(str)
# cache.transaction['price'] = cache.transaction['price'].astype(str)
# cache.transaction['sales_channel_id'] = cache.transaction['sales_channel_id'].astype(str)


# row['article_code'] = cache.transaction[['customer_id', 't_dat', 'article_code']].groupby(['customer_id', 't_dat'])['article_code'].apply(" ".join).reset_index()
# # row['article_code'] = row['article_code'][['customer_id', 't_dat', 'article_code']].groupby(['customer_id'])['article_code'].apply(" ".join).reset_index()
# row['price'] = cache.transaction[['customer_id', 't_dat', 'price']].groupby(['customer_id', 't_dat'])['price'].apply(" ".join).reset_index()
# row['price'] = row['price'][['customer_id', 't_dat', 'price']].groupby(['customer_id'])['price'].apply(" ".join).reset_index()
# row['sales_channel_id'] = cache.transaction[['customer_id', 't_dat', 'sales_channel_id']].groupby(['customer_id', 't_dat'])['sales_channel_id'].apply(" ".join).reset_index()
# row['sales_channel_id'] = row['sales_channel_id'][['customer_id', 't_dat', 'sales_channel_id']].groupby(['customer_id'])['sales_channel_id'].apply(" ".join).reset_index()
# row = [row[k] for k in row.keys()]
# cache.transaction = library.functools.reduce(lambda x,y: library.pandas.merge(left=x, right=y, on='customer_id', how='inner'), row)
# # cache.transaction['article_code'] = cache.transaction['article_code'].apply(lambda x: "1 " + x)
# # cache.transaction['price'] = cache.transaction['price'].apply(lambda x: "1 " + x)
# # cache.transaction['sales_channel_id'] = cache.transaction['sales_channel_id'].apply(lambda x: "1 " + x)
# cache.save(what=cache.transaction, file='transaction.csv', format='csv')

##  針對 customer 表進行前處理.
cache.customer = table.customer.copy()
cache.customer['FN'] = cache.customer['FN'].fillna(0.0)
cache.customer['Active'] = cache.customer['Active'].fillna(0.0)
cache.customer['club_member_status'] = cache.customer['club_member_status'].fillna("MISS")
cache.customer['fashion_news_frequency'] = cache.customer['fashion_news_frequency'].fillna("MISS")
cache.customer['age'] = cache.customer['age'].fillna(36.0)
cache.customer['club_member_status'] = feature.category.encode(cache.customer['club_member_status'], 0)
cache.customer['fashion_news_frequency'] = feature.category.encode(cache.customer['fashion_news_frequency'], 0)
cache.customer['age'] = cache.customer['age'] / 100
cache.customer['postal_code'] = feature.category.encode(cache.customer['postal_code'], 0)
cache.save(what=cache.customer, file='customer.csv', format='csv')

##  整合特徵.
cache.f1 = library.pandas.merge(left=cache.customer, right=cache.transaction, on='customer_id', how='inner')
cache.save(cache.f1, 'f1.csv', 'csv')

cache.save(cache.f1.head(2000), 'f1-sample.csv', 'csv')
# c = library.pandas.read_csv("./resource/preprocess/csv/customer.csv")
# c['postal_code'].nunique()



'''
##  根據用戶來建構特徵, 為了建構驗證資料, 會需要在交易紀錄表中動手腳.
##  刪除最後 step 次購買賞品的紀錄, 前面的歷史紀錄當作特徵, 第 step 時間購買的商品就是想要預測的值.
step = 1
for i in [i for i in range(step)]:

    table.transaction['index'] = range(len(table.transaction))
    skip = table.transaction.groupby(['customer_id']).last().reset_index()['index'].values
    table.transaction = table.transaction.drop(skip).reset_index(drop=True).drop(columns=['index'])
    pass

##  針對交易紀錄表, 根據用戶來建構基礎特徵.
level = {i:table.transaction[i].nunique() for i in table.transaction.keys()}
numerical = ['price']
categorical = [
    'sales_channel_id', 'product_group_name',
    'perceived_colour_value_id', 'perceived_colour_value_name', 
    'perceived_colour_master_id', 'perceived_colour_master_name', 
    'index_code', 'index_name', 'index_group_no', 'index_group_name', 
    'FN', 'Active', 'club_member_status', 'fashion_news_frequency'
]
for n in numerical:

    for c in categorical:
        
        cache.feature = feature.cross.statistic(table=table.transaction, key='customer_id', category=c, numeric=n)
        cache.save(what=cache.feature, file="{}_{}.csv".format(n, c), format='csv')
        pass

    pass

##  整合標記跟特徵.
for index, item in enumerate(library.os.listdir(cache.storage + '/csv/')):

    if(item=="article.csv") : continue
    item = library.pandas.read_csv(cache.storage + '/csv/{}'.format(item))
    if(index==0): cache.f1 = item
    if(index!=0): cache.f1 = library.pandas.merge(left=cache.f1, right=item, on="customer_id", how='outer')
    continue

cache.f1 = cache.f1.fillna(-1)
cache.f1 = library.pandas.merge(left=cache.f1, right=table.customer, on='customer_id', how='inner')
cache.save(what=cache.f1, file='f1.csv', format='csv')




# list(cache.f1.keys())
'''