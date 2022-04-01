
##  載入套件.
import feature
import library
import data

##  載入資料.
table = data.table(source='kaggle', mode='sample')
cache = feature.cache(storage='resource/preprocess(sample)')
pass

##  初步清理.
table.transaction = table.transaction
table.article["detail_desc"] = table.article["detail_desc"].fillna("<NA>")
table.customer['FN'] = table.customer['FN'].fillna(0.0)
table.customer['Active'] = table.customer['Active'].fillna(0.0)
table.customer['club_member_status'] = table.customer['club_member_status'].fillna("<NA>")
table.customer['fashion_news_frequency'] = table.customer['fashion_news_frequency'].fillna("<NA>")
table.customer['age'] = table.customer['age'].fillna(-1)
pass

##  針對 article 表進行前處理, 進行編碼.
cache.article = table.article.copy()
pass

target = 'article_code'
head = 10
cache.article[target] = feature.label.encode(cache.article['article_id']) + head
pass

loop = [
    'product_code', 'prod_name', "product_type_no", 'product_type_name',
    "product_group_name", "graphical_appearance_no", "graphical_appearance_name",
    "colour_group_code", 'colour_group_name', 'perceived_colour_value_id', 
    'perceived_colour_value_name', 'perceived_colour_master_id', 
    'perceived_colour_master_name', 'department_no', 'department_name', 
    'index_code', 'index_name', 'index_group_no', 'index_group_name', 
    'section_no', 'section_name', 'garment_group_no', 
    'garment_group_name', 'detail_desc'
]
for variable in loop:

    cache.article[variable] = feature.label.encode(cache.article[variable]) + head
    pass

cache.save(what=cache.article, file='article.csv', format='csv')
pass

##  針對 customer 表進行前處理, 以用戶為單位來建構特徵.
cache.customer = table.customer.copy()
pass

variable = 'club_member_status'
cache.customer[variable] = feature.label.encode(cache.customer[variable]) + head
pass

variable = 'fashion_news_frequency'
cache.customer[variable] = feature.label.encode(cache.customer[variable]) + head
pass

variable = 'postal_code'
cache.customer[variable] = feature.label.encode(cache.customer[variable]) + head
pass

cache.save(what=cache.customer, file='customer.csv', format='csv')
pass

##  針對 transaction 表進行前處理, 以用戶為單位來建構標記與特徵的序列.
cache.sequence = dict()
pass

variable = 't_dat'
cache.transaction = table.transaction.copy()
cache.transaction[variable] = feature.label.encode(cache.transaction[variable]) + head
cache.table = feature.sequence.flatten(table=cache.transaction.astype(str), key='customer_id', group=['customer_id'], variable=variable)
cache.sequence[variable] = cache.table
pass

variable = 'price'
cache.transaction = table.transaction.copy()
cache.table = feature.sequence.flatten(table=cache.transaction.astype(str), key='customer_id', group=['customer_id', 't_dat'], variable=variable)
cache.sequence[variable] = cache.table
pass

variable = 'sales_channel_id'
cache.transaction = table.transaction.copy()
cache.transaction[variable] = feature.label.encode(cache.transaction[variable]) + head
cache.table = feature.sequence.flatten(table=cache.transaction.astype(str), key='customer_id', group=['customer_id', 't_dat'], variable=variable)
cache.sequence[variable] = cache.table
pass

##  將 article 表的訊息納入前處理, 以用戶為單位來建構標記與特徵的序列.
loop = [
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
for variable in loop:

    print("start {} item in loop".format(variable))
    cache.transaction = table.transaction.copy()
    cache.transaction = library.pandas.merge(
        cache.transaction, 
        cache.article[["article_id", variable]], 
        on="article_id", how='inner'
    )
    cache.table = feature.sequence.flatten(table=cache.transaction.astype(str), key='customer_id', group=['customer_id', 't_dat'], variable=variable)
    cache.sequence[variable] = cache.table
    pass

merge = lambda x,y: library.pandas.merge(left=x, right=y, on='customer_id', how='inner')
cache.sequence = library.functools.reduce(merge, cache.sequence.values())
cache.save(what=cache.sequence, file='sequence.csv', format='csv')
pass

##  初步整合.
cache.f1 = library.pandas.merge(left=cache.customer, right=cache.sequence, on='customer_id', how='outer')
cache.save(what=cache.f1.dropna(), file='f1.csv', format='csv')
cache.save(what=cache.f1.fillna(""), file='f1.csv(global)', format='csv')
pass


