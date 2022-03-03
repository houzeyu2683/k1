
import pandas
import time
import os

root = './resource/'
class table:
    
    def __init__(self, source=None):

        if(source=='kaggle'):

            start = time.time()
            self.transaction = pandas.read_csv(os.path.join(root, source, 'csv', "transactions_train.csv"), dtype={'article_id':str})
            self.submission = pandas.read_csv(os.path.join(root, source, 'csv', "sample_submission.csv"))
            self.article = pandas.read_csv(os.path.join(root, source, 'csv', "articles.csv"))
            self.customer = pandas.read_csv(os.path.join(root, source, 'csv', "customers.csv"))
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

