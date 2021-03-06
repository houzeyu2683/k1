
import pandas
import time
import os
from sklearn import preprocessing

root = './resource/'
class table:
    
    def __init__(self, source=None, mode='default'):

        if(source=='kaggle'):

            ##  計算時間起點.
            start = time.time()

            ##  資料 transaction 表.
            path = os.path.join(root, source, 'csv', "transactions_train.csv")
            if(mode=='default'): self.transaction = pandas.read_csv(path, dtype={'article_id':str})
            if(mode=='sample'): self.transaction = pandas.read_csv(path, dtype={'article_id':str}, nrows=1000000).sample(100000, random_state=0).sort_values(by=['t_dat']).reset_index(drop=True)
            
            ##  資料 submission 表.
            path = os.path.join(root, source, 'csv', "sample_submission.csv")
            self.submission = pandas.read_csv(path)
            
            ##  資料 article 表. 
            path = os.path.join(root, source, 'csv', "articles.csv")
            self.article = pandas.read_csv(path, dtype={'article_id':str})

            ##  提前處理 customer 資料表.
            path = os.path.join(root, source, 'csv', "customers.csv")
            self.customer = pandas.read_csv(path)

            ##  計算時間終點.
            end = time.time()
            print('elapsed {} time'.format(end-start))
            pass

        if(source=='preprocess'):

            start = time.time()
            path = os.path.join(root, source, 'csv', 'f1.csv')
            self.f1 = pandas.read_csv(path, low_memory=False)
            end = time.time()
            print('elapsed {} time'.format(end-start))
            pass
        
        return

    pass

