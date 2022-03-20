
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
import random

class constant:

    version = '1.0.1'
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
        collection["i"]    = []
        collection["ii"]   = []
        collection["iii"]  = []
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            engine.prepare()
            collection['item'] += [engine.item]
            collection["i"] += [engine.handle(step=1)]
            collection["ii"] += [engine.handle(step=2)]
            collection["iii"] += [engine.handle(step=3)]
            # collection["iv"] += [engine.handle(step=4)]
            pass

        collection['item']     = pandas.concat(collection['item'],axis=1).transpose()
        collection['i']        = torch.stack(collection['i'], 0)
        collection['ii']       = torch.stack(collection['ii'], 1)
        # history = rnn.pad_sequence([h for h, _ in collection['iii']], batch_first=False, padding_value=0)
        # future  = rnn.pad_sequence([f for f, _ in collection['iii']], batch_first=False, padding_value=0)
        # collection['iii'] = history, future
        s = []
        for i in range(len(engine.loop)):
            
            h = rnn.pad_sequence([b[i][0] for b in collection['iii']], batch_first=False, padding_value=0)
            f = rnn.pad_sequence([b[i][1] for b in collection['iii']], batch_first=False, padding_value=0)
            s += [(h, f)]
            pass

        collection['iii'] = s
        return(collection)
    
    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        pass
    
    def prepare(self):

        self.blank = 1
        self.point = self.item['seq_len'] - self.blank
        self.loop = [
            "article_code", "sales_channel_id",
            "product_code", "prod_name", "product_type_no", 
            "product_type_name", "product_group_name", "graphical_appearance_no", 
            "graphical_appearance_name", "colour_group_code", "colour_group_name", 
            "perceived_colour_value_id", "perceived_colour_value_name", "perceived_colour_master_id", 
            "perceived_colour_master_name", "department_no", "department_name", 
            "index_code", "index_name", "index_group_no", 
            "index_group_name", "section_no", "section_name", 
            "garment_group_no", "garment_group_name", "detail_desc", 
            'price'
        ]
        self.knife = random.randint(0, self.point)
        return

    ##  Handle item.
    def handle(self, step=1):

        if(step==1):

            selection = ["FN", "Active", "club_member_status", "fashion_news_frequency", "age"]
            output = torch.tensor(self.item[selection]).type(torch.FloatTensor)
            pass

        if(step==2):

            selection = ['postal_code']
            output = torch.tensor(self.item[selection]).type(torch.LongTensor)
            pass

        # if(step==3):

        #     h = [float(i) for i in self.item['price'].split()[:self.knife]]
        #     f = [float(i) for i in self.item['price'].split()[self.knife:][:self.blank]]
        #     history = torch.tensor([0.0] + h).type(torch.FloatTensor)
        #     future  = torch.tensor([0.0] + f).type(torch.FloatTensor)
        #     output = history, future
        #     pass

        if(step==3):

            sequence = []
            for l in self.loop:

                h = [float(i) for i in self.item[l].split()[:self.knife]]
                f = [float(i) for i in self.item[l].split()[self.knife:][:self.blank]]
                if(l!='price'):

                    history = torch.tensor([1.0] + h).type(torch.LongTensor)
                    future = torch.tensor([2.0] + f).type(torch.LongTensor)
                    pass

                else:

                    history = torch.tensor([0.0] + h).type(torch.FloatTensor)
                    future  = torch.tensor([0.0] + f).type(torch.FloatTensor)
                    pass
                
                sequence += [(history, future)]
                pass

            output = sequence
            pass

        return(output)

    pass

        

