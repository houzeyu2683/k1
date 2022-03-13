
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
    article = pandas.read_csv("resource/preprocess/csv/article.csv", low_memory=False).drop(['article_id'], axis=1).to_numpy()

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
        collection['batch size'] = len(iteration)
        collection['mode'] = mode
        collection['item'] = []
        collection['y'] = []
        collection['x1'] = []
        collection['x2'] = []
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            collection['item'] += [engine.item]
            collection['y'] += [engine.y()]
            collection['x1'] += [engine.x1()]
            collection['x2'] += [engine.x2()]
            pass

        collection['item'] = pandas.concat(collection['item'],axis=1).transpose()
        collection['y']    = torch.stack(collection['y'], 0)
        collection['x1']   = torch.stack(collection['x1'], 0)
        collection['x2']   = connect(rnn.pad_sequence(collection['x2'], batch_first=False, padding_value=0))
        return(collection)
    
    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        pass
    
    def x1(self):

        vector = self.item.loc[(self.item.keys()!='article_code') & (self.item.keys()!='customer_id')]
        output = torch.tensor(vector).type(torch.FloatTensor)
        return(output)

    def x2(self):

        vector = [int(i) for i in self.item['article_code'].split()[:-1]]
        output = torch.tensor(vector, dtype=torch.int64)
        return(output)
        
    def y(self):

        vector = int(self.item['article_code'].split()[-1])
        output = torch.tensor(vector, dtype=torch.int64)
        return(output)

    pass

def connect(x="(length, batch)"):

    x = [constant.article[x[l,:], :] for l in range(len(x))]
    x = torch.tensor(numpy.stack(x, axis=0))
    return(x)

# item = dataset.train.__getitem__(2)
# item = item[["FN", "Active", "club_member_status", "fashion_news_frequency", "age", "postal_code"]]
# output = torch.tensor(vector, dtype=torch.float64)
# import torch
# [k for k in x.split()]
# [
# vocabulary['article']
# table.article
# class process:

#     def __init__(self, item=None, mode='train'):

#         self.item = item
#         self.mode = mode
#         pass

#     def user(self):

        
#         self.item['<column>'].type(torch.float32)
#         return
 
#     def image(self):

#         path = os.path.join(constant['image folder'], self.item['image'])
#         picture = PIL.Image.open(path).convert("RGB")
#         picture = zoom(picture, 256)
#         if(self.mode=='train'):
            
#             blueprint = [
#                 kit.RandomHorizontalFlip(p=0.5),
#                 kit.RandomRotation((-45, +45)),
#                 # kit.ColorJitter(brightness=(0,1), contrast=(0,1),saturation=(0,1), hue=(-0.5, 0.5)),
#                 kit.RandomCrop(constant['image size']),
#                 kit.ToTensor(),
#             ]
#             convert = kit.Compose(blueprint)
#             picture = convert(picture)
#             pass

#         else:
            
#             blueprint = [
#                 kit.CenterCrop(constant['image size']),
#                 kit.ToTensor(),
#             ]
#             convert = kit.Compose(blueprint)
#             picture = convert(picture)                    
#             pass
            
#         picture = picture.type(torch.float32)
#         return(picture)

#     def text(self):

#         index = self.vocabulary.encode(self.item['text'])
#         index = torch.tensor(index, dtype=torch.int64)
#         return(index)

#     pass

# '''
# 根據圖片最短邊進行縮放，使得最短邊縮放後超過一定的長度。
# '''
# def zoom(picture=None, boundary=None):

#     size = picture.size
#     scale = (boundary // size[numpy.argmin(size, axis=None)]) + 1
#     height, width = size[0]*scale, size[1]*scale
#     picture = picture.resize((height, width))
#     return(picture)

