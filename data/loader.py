
import pandas
import torch
import PIL.Image
import numpy
import os
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from torchvision import transforms as kit
from functools import partial
import json

class constant:

    version = '1.0.1'
    article = "resource/preprocess/csv/article.csv"
    pass

if(os.path.isfile(constant.article)):

    constant.article = pandas.read_csv(constant.article, low_memory=False).drop(['article_id'], axis=1).to_numpy()
    pass

def extend(x="article: (length, batch)"):

    x = [constant.article[x[l,:], :] for l in range(len(x))]
    x = torch.tensor(numpy.stack(x, axis=0))
    return(x)

class loader:

    def __init__(self, batch=32):

        self.batch = batch
        return
    
    def define(self, train=None, validation=None, test=None):

        ##  Train loader.
        if(train!=None):

            self.train = DataLoader(
                dataset=train, batch_size=self.batch, 
                shuffle=True , drop_last=True, 
                collate_fn=partial(self.collect, mode='train')
            )
            pass
        
        ##  Validation loader.
        if(validation!=None):

            self.validation = DataLoader(
                dataset=validation, batch_size=4, 
                shuffle=False , drop_last=False, 
                collate_fn=partial(self.collect, mode='validation')
            )
            pass

        ##  Test loader.
        if(test!=None):

            self.test = DataLoader(
                dataset=test, batch_size=4, 
                shuffle=False , drop_last=False, 
                collate_fn=partial(self.collect, mode='test')
            )
            pass

        return

    def collect(self, iteration, mode):
        
        collection = dict()
        collection['size'] = len(iteration)
        collection['mode'] = mode
        collection['item'] = []
        collection['x1'] = []
        collection['x2'] = []
        collection['x3'] = []
        collection['x4'] = []
        collection['x5'] = []
        collection['y'] = []
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            collection['item'] += [engine.item]
            collection['x1'] += [engine.x1()]
            collection['x2'] += [engine.x2()]
            collection['x3'] += [engine.x3()]
            collection['x4'] += [engine.x4()]
            collection['x5'] += [engine.x5()]
            collection['y'] += [engine.y()]
            pass

        collection['item']      = pandas.concat(collection['item'],axis=1).transpose()
        collection['x1']        = torch.stack(collection['x1'], 0)
        collection['x2']        = torch.stack(collection['x2'], 0)
        collection['x3']        = extend(rnn.pad_sequence(collection['x3'], batch_first=False, padding_value=0))
        collection['x4']        = rnn.pad_sequence(collection['x4'], batch_first=False, padding_value=0)
        collection['x5']        = rnn.pad_sequence(collection['x5'], batch_first=False, padding_value=0)
        collection['y']         = torch.stack(collection['y'], 0)
        return(collection)
    
    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        pass
    
    def x1(self):

        vector = self.item[['FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'age']]
        output = torch.tensor(vector).type(torch.FloatTensor)
        return(output)

    def x2(self):

        vector = self.item[["postal_code"]]
        output = torch.tensor(vector).type(torch.LongTensor)
        return(output)

    def x3(self):

        vector = [int(i) for i in self.item['article_code'].split()[:-1]]
        output = torch.tensor(vector).type(torch.LongTensor)
        return(output)

    def x4(self):

        vector = [float(i) for i in self.item['price'].split()[:-1]]
        output = torch.tensor(vector).type(torch.FloatTensor)
        return(output)

    def x5(self):

        vector = [int(i) for i in self.item['sales_channel_id'].split()[:-1]]
        output = torch.tensor(vector).type(torch.LongTensor)
        return(output)
        
    def y(self):

        order = self.item['article_code'].split()
        vector = [int(order[-1]), len(order), 1]
        output = torch.tensor(vector, dtype=torch.int64)
        return(output)

    pass



