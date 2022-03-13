
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
    # vocabulary = {'article':"resource/preprocess/json/article.json"}
    article = "resource/preprocess/csv/article.csv"
    pass

# class vocabulary:

#     if(os.path.isfile(constant.vocabulary['article'])):

#         with open(constant.vocabulary['article']) as paper: article = json.load(paper)
#         pass
    
#     pass


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
        collection['sequence'] = []
        collection['variable'] = []
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            collection['item'] += [engine.item]
            collection['sequence'] += [engine.sequence()]
            collection['variable'] += [engine.variable()]
            pass

        collection['item']     = pandas.concat(collection['item'],axis=1).transpose()
        collection['variable']  = torch.stack(collection['variable'], 0)
        # batch['image']  = torch.stack(batch['image'], 0)
        collection['sequence'] = rnn.pad_sequence(collection['sequence'], batch_first=False, padding_value=0)
        return(collection)
    
    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        pass
    
    def sequence(self):

        length = 12
        token = [vocabulary.article[k] for k in self.item['sequence'].split()]
        difference = len(token)-length
        if(difference<0): token = token + ([0]*(-difference))
        token  = token[0:length]
        output = torch.tensor(token, dtype=torch.int64)
        return(output)
        
    def variable(self):

        vector = self.item[["FN", "Active", "club_member_status", "fashion_news_frequency", "age", "postal_code"]]
        output = torch.tensor(vector).type(torch.FloatTensor)
        return(output)

    pass




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