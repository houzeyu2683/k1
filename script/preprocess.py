
##
import feature
import library
import data

##  Load all data table.
table = data.table(source='kaggle')

##  宣告暫存物件.
cache = data.cache(storage='resource/preprocess')

##  商品資訊資料表與交易紀錄表進行合併, 擴充交易資料表.
table.transaction = library.pandas.merge(table.transaction, table.article, on="article_id", how='inner')
table.transaction = library.pandas.merge(table.transaction, table.customer, on="customer_id", how='inner')
level = {i:table.transaction[i].nunique() for i in table.transaction.keys()}

##  針對交易紀錄表, 根據用戶來建構基礎特徵, 有些類別的數量實在過多, 將其當作連續型變數來進行妥協處理.
numerical = ['price']
categorical = [
    "sales_channel_id", "product_type_no", 
    "product_group_name", "graphical_appearance_no", 
    "graphical_appearance_name", "colour_group_code", "colour_group_name", 
    "perceived_colour_value_id", "perceived_colour_value_name",
    "perceived_colour_master_id", "perceived_colour_master_name",
    "", "department_name", "index_code", "index_name",
    "index_group_no", "index_group_name", "section_no", "section_name", 
    "garment_group_no", "garment_group_name", "detail_desc"
]
for n in numerical:

    for c in categorical:

        cache.feature = feature.cross.statistic(table=table.transaction, key='customer_id', category=c, numeric=n)
        cache.save(what=cache.feature, file="{}_{}.csv".format(n, c), format='csv')
        pass

    pass




table.transaction





##  建構標記資料, 根據提交表的格式, 以用戶當作 row 來建構對應的標記.
cache.label = table.transaction[['customer_id', 't_dat', 'article_id']].groupby(['customer_id', 't_dat'])['article_id'].apply(" ".join).reset_index()
cache.label = cache.label.pivot_table(values='article_id', index='customer_id', columns='t_dat', aggfunc='first')
cache.label = cache.label.rename_axis(None, axis=1).reset_index()
cache.label['sequence'] = cache.label.fillna("").iloc[:,1:].apply(lambda x: " ".join(" ".join(x).split()), 1)
cache.label = cache.label[['customer_id', 'sequence']]
cache.label = library.pandas.merge(cache.label, table.customer[['customer_id']], on="customer_id", how='outer')
cache.label['sequence'] = cache.label['sequence'].fillna('<beginner>')
cache.save(what=cache.label, file='label.csv', format='csv')

##  用戶資料處理, 以用戶當作 row 來建構對應的基礎特徵.
cache.user = table.customer.copy()
cache.user['FN'] = cache.user['FN'].fillna(0.0)
cache.user['Active'] = cache.user['Active'].fillna(0.0)
encoder = library.preprocessing.LabelEncoder()
cache.user['club_member_status'] = cache.user['club_member_status'].fillna('unknown')
encoder.fit(cache.user['club_member_status'].unique())
cache.user['club_member_status'] = encoder.transform(cache.user['club_member_status'])
encoder = library.preprocessing.LabelEncoder()
cache.user['fashion_news_frequency'] = cache.user['fashion_news_frequency'].fillna('unknown')
encoder.fit(cache.user['fashion_news_frequency'].unique())
cache.user['fashion_news_frequency'] = encoder.transform(cache.user['fashion_news_frequency'])
cache.user['age'] = cache.user['age'].fillna(36.0)
cache.user['age'] = round(cache.user['age'] / cache.user['age'].max(), 2)
encoder = library.preprocessing.LabelEncoder()
cache.user['postal_code'] = cache.user['postal_code'].fillna('unknown')
encoder.fit(cache.user['postal_code'].unique())
cache.user['postal_code'] = encoder.transform(cache.user['postal_code'])
cache.save(what=cache.user, file='user.csv', format='csv')

##  整合上述流程至一張表, 用以訓練模型使用.
cache.f1 = library.pandas.merge(cache.user, cache.label, on="customer_id", how='outer')
cache.save(what=cache.f1, file='f1.csv', format='csv')

##  針對需要預測的商品建構商品編碼表, 將商品編號轉換到整數, 用於 embedding 使用.
cache.vocabulary = dict()
cache.vocabulary['article'] = {key:value for value, key in enumerate(['<beginner>'] + list(table.article['article_id']))}
cache.save(what=cache.vocabulary['article'], file='article.json', format='json')


# 0663713001
# cache.label
# table.article['article_id'].min()
# table.transaction['article_id'].min()
# # cache.f1.keys()

# x = set.intersection(set(table.article['article_id']), set(table.transaction['article_id']))
# len(x)
# table.article['article_id'].nunique()
# table.transaction['article_id'].nunique()
# cache.label['sequence'].min()

# table.article['article_id'].max()

# cache.user = pandas.merge(cache.user, table.customer, on="customer_id", how='outer')
# cache.save(what=cache.user, file='user.csv', format='csv')
# pass

# table = data.table(source='preprocess')
# cache = data.cache(storage='resource/preprocess/csv')
# cache.user = table.user[['sequence', 'FN', "Active", 'club_member_status', "fashion_news_frequency", 'age', 'postal_code']]
# print("column [FN] missing value {} %".format(cache.user['FN'].isna().sum() / len(cache.user)))
# print("column [Active] missing value {} %".format(cache.user['Active'].isna().sum() / len(cache.user)))
# print("column [club_member_status] missing value {} %".format(cache.user['club_member_status'].isna().sum() / len(cache.user)))
# print("column [fashion_news_frequency] missing value {} %".format(cache.user['fashion_news_frequency'].isna().sum() / len(cache.user)))
# print("column [age] missing value {} %".format(cache.user['age'].isna().sum() / len(cache.user)))
# print("column [postal_code] missing value {} %".format(cache.user['postal_code'].isna().sum() / len(cache.user)))




