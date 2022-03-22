
import pandas
import time
import os
from sklearn import preprocessing

root = './resource/'
class table:
    
    def __init__(self, source=None, sample=False):

        if(source=='kaggle'):

            ##  計算時間起點.
            start = time.time()

            ##  資料 transaction 表.
            path = os.path.join(root, source, 'csv', "transactions_train.csv")
            self.transaction = pandas.read_csv(path, dtype={'article_id':str}) if(not sample) else pandas.read_csv(path, dtype={'article_id':str}, nrows=100000)

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

        else:

            start = time.time()
            path = os.path.join(root, source, 'csv', 'f1.csv')
            self.f1 = pandas.read_csv(path, low_memory=False)
            # self.user = pandas.read_csv(os.path.join(root, source, 'csv', 'user.csv'), low_memory=False)
            # self.target = pandas.read_csv(os.path.join(root, source, 'csv', 'target.csv'), low_memory=False)
            end = time.time()
            print('elapsed {} time'.format(end-start))
            
        
        return

    pass

