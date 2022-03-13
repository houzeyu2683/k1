
##
import feature
import library
import data

##  宣告暫存物件.
cache = data.cache(storage='resource/preprocess')

##  Load all data table.
table = data.table(source='kaggle')

##
cache.save(what=table.article, file='article.csv', format='csv')

##  商品資訊資料表與交易紀錄表進行合併, 擴充交易資料表.
table.transaction = library.pandas.merge(table.transaction, table.article, on="article_id", how='inner')
table.transaction = library.pandas.merge(table.transaction, table.customer, on="customer_id", how='inner')

##  建構標記資料, 根據提交表的格式, 以用戶當作 row 來建構對應的標記.
cache.label = table.transaction[['customer_id', 't_dat', 'article_code']].groupby(['customer_id', 't_dat'])['article_code'].apply(" ".join).reset_index()
cache.label = cache.label.groupby(['customer_id'])['article_code'].apply(" ".join).reset_index()
cache.label['article_code'] = ["1 " + i for i in cache.label['article_code']]
cache.save(what=cache.label, file='label.csv', format='csv')

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





