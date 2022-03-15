
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

# class constant:

#     version = '1.0.1'
#     article = pandas.read_csv("resource/preprocess/csv/article.csv", low_memory=False).drop(['article_id'], axis=1).to_numpy()

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
                dataset=validation, batch_size=self.batch, 
                shuffle=False , drop_last=False, 
                collate_fn=partial(self.collect, mode='validation')
            )
            pass

        ##  Test loader.
        if(test!=None):

            self.test = DataLoader(
                dataset=test, batch_size=self.batch, 
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
        collection['target'] = []
        collection['row'] = []
        collection['sequence'] = []
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            collection['item'] += [engine.item]
            collection['target'] += [engine.target()]
            collection['row'] += [engine.row()]
            collection['sequence'] += [engine.sequence()]
            pass

        collection['item'] = pandas.concat(collection['item'],axis=1).transpose()
        collection['target']    = torch.stack(collection['target'], 0)
        collection['row']   = torch.stack(collection['row'], 0)
        collection['sequence']   = connect(rnn.pad_sequence(collection['sequence'], batch_first=False, padding_value=0))
        return(collection)
    
    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        pass
    
    def row(self):

        vector = self.item.loc[(self.item.keys()!='article_code') & (self.item.keys()!='customer_id')]
        output = torch.tensor(vector).type(torch.FloatTensor)
        return(output)

    def sequence(self):

        vector = [int(i) for i in self.item['article_code'].split()[:-1]]
        output = torch.tensor(vector, dtype=torch.int64)
        return(output)
        
    def target(self):

        vector = int(self.item['article_code'].split()[-1])
        output = torch.tensor(vector, dtype=torch.int64)
        return(output)

    pass

def connect(x="(length, batch)"):

    x = [constant.article[x[l,:], :] for l in range(len(x))]
    x = torch.tensor(numpy.stack(x, axis=0))
    return(x)

