
import pandas
import time
import os
from sklearn import preprocessing

root = './resource/'
class table:
    
    def __init__(self, source=None):

        if(source=='kaggle'):

            start = time.time()

            ##  
            self.transaction = pandas.read_csv(os.path.join(root, source, 'csv', "transactions_train.csv"), dtype={'article_id':str})
            encoder = preprocessing.LabelEncoder()
            encoder.fit(self.transaction['sales_channel_id'].unique())
            self.transaction['sales_channel_id'] = encoder.transform(self.transaction["sales_channel_id"])

            ##
            self.submission = pandas.read_csv(os.path.join(root, source, 'csv', "sample_submission.csv"))
            
            ##
            self.article = pandas.read_csv(os.path.join(root, source, 'csv', "articles.csv"), dtype={'article_id':str})
            encoder = preprocessing.LabelEncoder()
            encoder.fit(self.article['article_id'].unique())
            self.article['target'] = encoder.transform(self.article["article_id"])
            self.article['detail_desc'] = self.article['detail_desc'].fillna("detail_desc unknown")
            loop = [
                "product_code", "prod_name", "product_type_no", "product_type_name", "product_group_name", 
                "graphical_appearance_no", "graphical_appearance_name", "colour_group_code", "colour_group_name", "perceived_colour_value_id", 
                "perceived_colour_value_name", "perceived_colour_master_id", "perceived_colour_master_name", 
                "department_no", "department_name", "index_code", "index_name", "index_group_no", "index_group_name", 
                "section_no", "section_name", "garment_group_no", "garment_group_name", "detail_desc"
            ]
            for i in loop:

                encoder = preprocessing.LabelEncoder()
                encoder.fit(self.article[i].unique())
                self.article[i] = encoder.transform(self.article[i])
                pass
            
            ##
            self.customer = pandas.read_csv(os.path.join(root, source, 'csv', "customers.csv"))
            self.customer['FN'] = self.customer['FN'].fillna(0.0)
            self.customer['Active'] = self.customer['Active'].fillna(0.0)
            self.customer['club_member_status'] = self.customer['club_member_status'].fillna('club_member_status unknown')
            self.customer['fashion_news_frequency'] = self.customer['fashion_news_frequency'].fillna('fashion_news_frequency unknown')
            self.customer['age'] = (self.customer['age'].fillna(36.0) / 100).round(2)
            self.customer['postal_code'] = self.customer['postal_code'].fillna('postal_code unknown')
            loop = ["club_member_status", "fashion_news_frequency", "postal_code"]
            for i in loop:

                encoder = preprocessing.LabelEncoder()
                encoder.fit(self.customer[i].unique())
                self.customer[i] = encoder.transform(self.customer[i])
                pass

            end = time.time()
            print('elapsed {} time'.format(end-start))
            return

        if(source=='preprocess'):

            start = time.time()
            self.f1 = pandas.read_csv(os.path.join(root, source, 'csv', 'f1.csv'), low_memory=False)
            # self.user = pandas.read_csv(os.path.join(root, source, 'csv', 'user.csv'), low_memory=False)
            # self.target = pandas.read_csv(os.path.join(root, source, 'csv', 'target.csv'), low_memory=False)
            end = time.time()
            print('elapsed {} time'.format(end-start))
            return
        
        pass

    pass

