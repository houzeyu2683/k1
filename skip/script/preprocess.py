
from library import *
import data

##  Load all data table.
table = data.table(source='kaggle')

'''
建構特徵資料, 根據提交表的格式, 以用戶當作 row 來建構對應的特徵.
發現有部份用戶過去沒有任何交易紀錄, 但是卻出現在提交表, 可以合理假設比賽是根據時間來切分測試集, 
表示測試集中有部份用戶是新加入的, 因此過去沒有交易訊息, 屬於冷啟動的預測項目, 需要額外處理.
'''

cache = data.cache(storage='resource/preprocess/csv')
cache.user = table.transaction.groupby(['customer_id', 't_dat'])['article_id'].apply(" ".join).reset_index()
cache.user = cache.user.pivot_table(values='article_id', index='customer_id', columns='t_dat', aggfunc='first')
cache.user = cache.user.rename_axis(None, axis=1).reset_index()
cache.user['sequence'] = cache.user.fillna("").iloc[:,1:].apply(lambda x: " ".join(" ".join(x).split()), 1)
cache.user = pandas.merge(cache.user, table.customer, on="customer_id", how='outer')
cache.save(what=cache.user, file='user.csv', format='csv')
pass

table = data.table(source='preprocess')
cache = data.cache(storage='resource/preprocess/csv')
cache.user = table.user[['sequence', 'FN', "Active", 'club_member_status', "fashion_news_frequency", 'age', 'postal_code']]
print("column [FN] missing value {} %".format(cache.user['FN'].isna().sum() / len(cache.user)))
print("column [Active] missing value {} %".format(cache.user['Active'].isna().sum() / len(cache.user)))
print("column [club_member_status] missing value {} %".format(cache.user['club_member_status'].isna().sum() / len(cache.user)))
print("column [fashion_news_frequency] missing value {} %".format(cache.user['fashion_news_frequency'].isna().sum() / len(cache.user)))
print("column [age] missing value {} %".format(cache.user['age'].isna().sum() / len(cache.user)))
print("column [postal_code] missing value {} %".format(cache.user['postal_code'].isna().sum() / len(cache.user)))




