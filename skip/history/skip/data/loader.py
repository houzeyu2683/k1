
import pandas
import torch
import PIL.Image
import numpy
import os
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from torchvision import transforms as kit
from functools import partial

class constant:

    version = '1.0.0'
    pass

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

    def collect(self, data, mode):
        
        collection = dict()
        collection['batch size'] = len(data)
        collection['mode'] = mode
        for item in data:
            
            engine = process(item=item, vocabulary=self.vocabulary, mode=mode)
            batch['item'] += [engine.item]
            batch['image'] += [engine.image()]
            batch['text'] += [engine.text()]
            pass

        batch['item']   = pandas.concat(batch['item'],axis=1).transpose()
        batch['image']  = torch.stack(batch['image'], 0)
        batch['text']   = rnn.pad_sequence(batch['text'], batch_first=False, padding_value=self.vocabulary.index['<padding>'])
        batch['size']   = len(group)
        batch['length'] = len(batch['text'])
        return(batch)
    
    pass

class process:

    def __init__(self, item=None, mode='train'):

        self.item = item
        self.mode = mode
        pass

    def user(self):

        
        self.item['<column>'].type(torch.float32)
        return
 
    def image(self):

        path = os.path.join(constant['image folder'], self.item['image'])
        picture = PIL.Image.open(path).convert("RGB")
        picture = zoom(picture, 256)
        if(self.mode=='train'):
            
            blueprint = [
                kit.RandomHorizontalFlip(p=0.5),
                kit.RandomRotation((-45, +45)),
                # kit.ColorJitter(brightness=(0,1), contrast=(0,1),saturation=(0,1), hue=(-0.5, 0.5)),
                kit.RandomCrop(constant['image size']),
                kit.ToTensor(),
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture)
            pass

        else:
            
            blueprint = [
                kit.CenterCrop(constant['image size']),
                kit.ToTensor(),
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture)                    
            pass
            
        picture = picture.type(torch.float32)
        return(picture)

    def text(self):

        index = self.vocabulary.encode(self.item['text'])
        index = torch.tensor(index, dtype=torch.int64)
        return(index)

    pass

# '''
# 根據圖片最短邊進行縮放，使得最短邊縮放後超過一定的長度。
# '''
# def zoom(picture=None, boundary=None):

#     size = picture.size
#     scale = (boundary // size[numpy.argmin(size, axis=None)]) + 1
#     height, width = size[0]*scale, size[1]*scale
#     picture = picture.resize((height, width))
#     return(picture)