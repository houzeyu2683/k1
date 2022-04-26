
##  The packages
import os
import numpy
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from functools import partial
import random

def pad(x='list of tensor', value=0, right=True):
    
    if(right): 
        
        y = rnn.pad_sequence(x, batch_first=False, padding_value=value)
        pass

    else:

        x = [i.flip(dims=[0]) for i in x]
        x = rnn.pad_sequence(x, batch_first=False, padding_value=value)
        y = x.flip(dims=[0])        
        pass

    "length, batch"
    return(y)

##  Data loader.
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
        
        batch = dict()
        batch['size'] = len(iteration)
        batch['mode'] = mode
        batch['boundary'] = []
        batch['item'] = []
        batch['bottom'] = None
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            engine.prepare()
            batch['boundary'] += [engine.boundary]
            batch['item'] += [pandas.DataFrame(engine.item).transpose()]
            pass

            ##  Vector.
            vector = engine.handle(step="vector")
            for k in vector:

                if(k in batch): batch[k] += [vector[k]]
                if(k not in batch): batch[k] = [vector[k]]
                pass

            ##  Sequence
            sequence = engine.handle(step='sequence')
            for k in sequence:

                h, f = 'history', 'future'
                if(k in batch): batch[k][h] += [sequence[k][h]]
                if(k in batch): batch[k][f] += [sequence[k][f]]
                if(k not in batch): batch[k] = {h:[sequence[k][h]], f:[sequence[k][f]]}
                pass
            
            pass
        
        batch['item']  = pandas.concat(batch['item'])
        pass
        
        ##  Vector.
        batch['FN'] = torch.cat(batch["FN"], 0).type(torch.FloatTensor) # (batch, 1)
        batch['Active'] = torch.cat(batch["Active"], 0).type(torch.FloatTensor) # (batch, 1)
        batch['age'] = torch.cat(batch['age'], 0).type(torch.FloatTensor) # (batch, 1)
        batch['club_member_status'] = torch.cat(batch["club_member_status"], 1).type(torch.LongTensor) # (1, batch)
        batch["fashion_news_frequency"] = torch.cat(batch["fashion_news_frequency"], 1).type(torch.LongTensor) # (1, batch)
        batch["postal_code"] = torch.cat(batch["postal_code"], 1).type(torch.LongTensor) # (1, batch)
        pass
        
        # ##  Sequence.
        h, f = 'history', 'future'
        # batch['price'][h] = rnn.pad_sequence(batch['price'][h], batch_first=False, padding_value=0).squeeze(-1).type(torch.FloatTensor)
        # batch['price'][f] = rnn.pad_sequence(batch['price'][f], batch_first=False, padding_value=0).squeeze(-1).type(torch.FloatTensor)
        batch['price'][h] = pad(batch['price'][h], value=0, right=False).type(torch.FloatTensor) # (length, batch, 1)
        batch['price'][f] = pad(batch['price'][f], value=0, right=False).type(torch.FloatTensor) # (length, batch, 1)

        key = [
            'sales_channel_id', 'product_code', 
            'prod_name', 'product_type_no', 'product_type_name', 
            'product_group_name', 'graphical_appearance_no', 
            'graphical_appearance_name', 'colour_group_code', 
            'colour_group_name', 'perceived_colour_value_id', 
            'perceived_colour_value_name', 'perceived_colour_master_id', 
            'perceived_colour_master_name', 'department_no', 
            'department_name', 'index_code', 'index_name', 
            'index_group_no', 'index_group_name', 'section_no', 
            'section_name', 'garment_group_no', 'garment_group_name', 
            'detail_desc', 'article_label'
        ]
        for k in key:

            batch[k][h] = rnn.pad_sequence(batch[k][h], batch_first=False, padding_value=0).squeeze(-1).type(torch.LongTensor)
            batch[k][f] = rnn.pad_sequence(batch[k][f], batch_first=False, padding_value=0).squeeze(-1).type(torch.LongTensor)
            pass

        return(batch)

    pass

##  Item process. 
class process:
    
    def __init__(self, item=None, mode=None):
    
        self.item = item.copy()
        self.mode = mode
        return
    
    def prepare(self):
        
        # print(self.item['length'])
        limit = float(self.item['length']) - 1
        self.boundary = random.randint(0, limit)
        return
    
    def handle(self, step=''):
    
        output = dict()
        pass

        if(step=="vector"):
            
            key = ["FN", "Active", "age", "club_member_status", "fashion_news_frequency", "postal_code"]
            for k in key:
    
                v = float(self.item[k])
                output[k] = torch.tensor([v]).unsqueeze(1)
                pass

            pass
    
        if(step=='sequence'):
    
            top = 12
            key = [
                'price', 'sales_channel_id',
                'product_code', 'prod_name', 'product_type_no',
                'product_type_name', 'product_group_name', 'graphical_appearance_no',
                'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
                'perceived_colour_value_id', 'perceived_colour_value_name',
                'perceived_colour_master_id', 'perceived_colour_master_name',
                'department_no', 'department_name', 'index_code', 'index_name',
                'index_group_no', 'index_group_name', 'section_no', 'section_name',
                'garment_group_no', 'garment_group_name', 'detail_desc',
                'article_label'
            ]
            for k in key:

                convert = lambda x: torch.tensor(x).unsqueeze(1)
                period = [float(i) for i in self.item[k].split()]
                history, future = period[:self.boundary], period[self.boundary:][:top]
                if(history==[]): history += [0.0]
                future  = future
                output[k] = {"history":convert(history), 'future':convert(future)}
                pass
            
            pass

        return(output)

    pass
