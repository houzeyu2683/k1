
import pandas
import time
import os
from sklearn import preprocessing

root = './resource/'
class table:
    
    def __init__(self, source=None):

        if(source=='kaggle'):

            ##  計算時間起點.
            start = time.time()

            ##  資料 transaction 表.
            self.transaction = pandas.read_csv(os.path.join(root, source, 'csv', "transactions_train.csv"), dtype={'article_id':str})

            ##  資料 submission 表.
            self.submission = pandas.read_csv(os.path.join(root, source, 'csv', "sample_submission.csv"))
            
            ##  資料 article 表. 
            self.article = pandas.read_csv(os.path.join(root, source, 'csv', "articles.csv"), dtype={'article_id':str})

            ##  提前處理 customer 資料表.
            self.customer = pandas.read_csv(os.path.join(root, source, 'csv', "customers.csv"))

            ##  計算時間終點.
            end = time.time()
            print('elapsed {} time'.format(end-start))
            return

        # if(source=='kaggle'):

        #     start = time.time()

        #     ##  提前處理 transaction 資料表.
        #     self.transaction = pandas.read_csv(os.path.join(root, source, 'csv', "transactions_train.csv"), dtype={'article_id':str}, nrows=100000)
        #     encoder = preprocessing.LabelEncoder()
        #     encoder.fit(self.transaction['sales_channel_id'].unique())
        #     self.transaction['sales_channel_id'] = encoder.transform(self.transaction["sales_channel_id"])

        #     ##  提前處理 submission 資料表.
        #     self.submission = pandas.read_csv(os.path.join(root, source, 'csv', "sample_submission.csv"))
            
        #     ##  提前處理 article 資料表, 編碼都保留 <start> 以及 <padding> 給 0 跟 1 來使用. 
        #     self.article = pandas.read_csv(os.path.join(root, source, 'csv', "articles.csv"), dtype={'article_id':str})
        #     encoder = preprocessing.LabelEncoder()
        #     encoder.fit(self.article['article_id'].unique())
        #     self.article['article_code'] = encoder.transform(self.article["article_id"]) + 2
        #     self.article['article_code'] = self.article['article_code'].astype(str)
        #     self.article['detail_desc'] = self.article['detail_desc'].fillna("unknown")
        #     loop = [
        #         "product_code", "prod_name", "product_type_no", "product_type_name", "product_group_name", 
        #         "graphical_appearance_no", "graphical_appearance_name", 
        #         "colour_group_code", "colour_group_name", 
        #         "perceived_colour_value_id", "perceived_colour_value_name", "perceived_colour_master_id", "perceived_colour_master_name", 
        #         "department_no", "department_name", 
        #         "index_code", "index_name", "index_group_no", "index_group_name", 
        #         "section_no", "section_name", 
        #         "garment_group_no", "garment_group_name", 
        #         "detail_desc"
        #     ]
        #     for i in loop:

        #         encoder = preprocessing.LabelEncoder()
        #         encoder.fit(self.article[i].unique())
        #         self.article[i] = encoder.transform(self.article[i]) + 2
        #         pass
            
        #     ##
        #     x = {i:[0] for i in loop}
        #     x.update({"article_id": ["<padding>"], "article_code": ["0"]})
        #     x = pandas.DataFrame(x)
        #     self.article  = pandas.concat([self.article, x]).copy()
        #     x = {i:[1] for i in loop}
        #     x.update({"article_id": ["<start>"], "article_code": ["1"]})
        #     x = pandas.DataFrame(x)
        #     self.article  = pandas.concat([self.article, x]).copy()

        #     ##  提前處理 customer 資料表.
        #     self.customer = pandas.read_csv(os.path.join(root, source, 'csv', "customers.csv"))
        #     self.customer['FN'] = self.customer['FN'].fillna(0.0)
        #     self.customer['Active'] = self.customer['Active'].fillna(0.0)
        #     self.customer['club_member_status'] = self.customer['club_member_status'].fillna('club_member_status unknown')
        #     self.customer['fashion_news_frequency'] = self.customer['fashion_news_frequency'].fillna('fashion_news_frequency unknown')
        #     self.customer['age'] = (self.customer['age'].fillna(36.0) / 100).round(2)
        #     self.customer['postal_code'] = self.customer['postal_code'].fillna('postal_code unknown')
        #     loop = ["club_member_status", "fashion_news_frequency", "postal_code"]
        #     for i in loop:

        #         encoder = preprocessing.LabelEncoder()
        #         encoder.fit(self.customer[i].unique())
        #         self.customer[i] = encoder.transform(self.customer[i]) 
        #         pass

        #     end = time.time()
        #     print('elapsed {} time'.format(end-start))
        #     return

        if(source=='preprocess'):

            start = time.time()
            self.f1 = pandas.read_csv(os.path.join(root, source, 'csv', 'f1.csv'), nrows=100000, low_memory=False)
            # self.user = pandas.read_csv(os.path.join(root, source, 'csv', 'user.csv'), low_memory=False)
            # self.target = pandas.read_csv(os.path.join(root, source, 'csv', 'target.csv'), low_memory=False)
            end = time.time()
            print('elapsed {} time'.format(end-start))
            return
        
        pass

    pass

