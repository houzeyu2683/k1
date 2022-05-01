
##  The packages
import os
import numpy
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from functools import partial
import random

schedule = {
    'customer_id':{'column':'customer_id', "type":str}, 
    'length':{"column":'length', "type":int}, 
    'FN':{"column":"p1", "type":torch.FloatTensor}, 
    'Active':{"column":'p2', "type":torch.FloatTensor}, 
    'club_member_status':{"column":"p3", "type":torch.LongTensor},
    'fashion_news_frequency':{"column":'p4', "type":torch.LongTensor}, 
    'age':{"column":'p5', "type":torch.FloatTensor}, 
    'postal_code':{"column":'p6', "type":torch.LongTensor}, 
    'product_code':{"column":'b1', "type":torch.LongTensor}, 
    'prod_name':{"column":'b2', "type":torch.LongTensor}, 
    'product_type_no':{"column":'b3', "type":torch.LongTensor},
    'product_type_name':{"column":'b4', "type":torch.LongTensor}, 
    'product_group_name':{"column":'b5', "type":torch.LongTensor}, 
    'graphical_appearance_no':{"column":'b6', "type":torch.LongTensor},
    'graphical_appearance_name':{"column":"b7", "type":torch.LongTensor}, 
    'colour_group_code':{"column":"b8", "type":torch.LongTensor},
    'colour_group_name':{"column":"b9", "type":torch.LongTensor},
    'perceived_colour_value_id':{"column":'b10', "type":torch.LongTensor}, 
    'perceived_colour_value_name':{"column":"b11", "type":torch.LongTensor},
    'perceived_colour_master_id':{"column":"b12", "type":torch.LongTensor}, 
    'perceived_colour_master_name':{"column":"b13", "type":torch.LongTensor},
    'department_no':{"column":'b14', "type":torch.LongTensor}, 
    'department_name':{"column":"b15", "type":torch.LongTensor}, 
    'index_code':{"column":"b16", "type":torch.LongTensor}, 
    'index_name':{"column":"b17", "type":torch.LongTensor},
    'index_group_no':{"column":"b18", "type":torch.LongTensor}, 
    'index_group_name':{"column":"b19", "type":torch.LongTensor}, 
    'section_no':{"column":"b20", "type":torch.LongTensor}, 
    'section_name':{"column":"b21", "type":torch.LongTensor},
    'garment_group_no':{"column":"b22", "type":torch.LongTensor}, 
    'garment_group_name':{"column":"b23", "type":torch.LongTensor}, 
    'detail_desc':{"column":"b24", "type":torch.LongTensor},
    'article_label':{"column":"b25", "type":torch.LongTensor},
    'price':{"column":'r1', "type":torch.FloatTensor},
    'sales_channel_id':{"column":'r2', "type":torch.LongTensor} 
}

class process:
    
    def __init__(self, item=None, mode='train'):
    
        self.item = item.copy()
        self.mode = mode
        return
    
    def handle(self):

        if(self.mode in ['train', 'validation']):

            output = dict()
            for origin, argument in schedule.items():

                if(origin in ['customer_id', 'length']): 
                    
                    x = self.item[[argument['column']]]
                    output[argument['column']] = x.astype(argument['type']).item()
                    continue
                
                if('p' in argument['column']):

                    x = torch.tensor([[float(self.item[argument['column']])]])
                    x = x.type(argument['type'])
                    output[argument['column']] = x
                    continue

                if('b' in argument['column']):
    
                    x = [float(i) for i in self.item[argument['column']].split()]
                    x = [1.0] + x
                    x = torch.cat([torch.tensor([[i]]) for i in x], dim=0)
                    x = x.type(argument['type'])
                    h, f = x[:-1], x[1:]
                    output[argument['column']] = h, f
                    continue

                if('r' in argument['column']):
        
                    x = [float(i) for i in self.item[argument['column']].split()]
                    x = [1.0] + x
                    x = torch.cat([torch.tensor([[i]]) for i in x], dim=0)
                    x = x.type(argument['type'])
                    h, f = x[:-1], x[1:]
                    output[argument['column']] = h, f
                    continue

                continue
            
            pass

        return(output)

def pad(sequence='list of tensor', value=0, right=True):
    
    if(right): 
        
        tensor = rnn.pad_sequence(sequence, batch_first=False, padding_value=value)
        pass

    else:

        sequence = [i.flip(dims=[0]) for i in sequence]
        sequence = rnn.pad_sequence(sequence, batch_first=False, padding_value=value)
        tensor = sequence.flip(dims=[0])        
        pass

    return(tensor)

class loader:

    def __init__(self, batch=32, device='cpu'):

        self.batch = batch
        self.device = device
        return
    
    def define(self, train=None, validation=None, test=None):

        if(train!=None):

            self.train = DataLoader(
                dataset=train, batch_size=self.batch, 
                shuffle=True , drop_last=True, 
                collate_fn=partial(self.collect, mode='train')
            )
            pass
        
        if(validation!=None):

            self.validation = DataLoader(
                dataset=validation, batch_size=4, 
                shuffle=False , drop_last=False, 
                collate_fn=partial(self.collect, mode='validation')
            )
            pass

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
        batch.update({'size':len(iteration)})
        batch.update({'mode':mode})
        batch.update({'item':pandas.concat(iteration, axis=1).transpose()})
        pass
    
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            feature = engine.handle()
            for key, value in feature.items():

                batch[key] = batch[key] + [value] if(key in batch) else [value]
                continue
            
            continue
        
        for _, argument in schedule.items():

            if(argument['column'] in ['p1', 'p2', 'p5']):

                x = torch.cat(batch[argument['column']], dim=0)
                batch[argument['column']] = x.to(self.device)
                continue

            if(argument['column'] in ['p3', 'p4', 'p6']):

                x = torch.cat(batch[argument['column']], dim=0).squeeze(1)
                batch[argument['column']] = x.to(self.device)
                continue

            if(argument['column'] in ['r1']):
                
                h = [h for h, _ in batch[argument['column']]]
                f = [f for _, f in batch[argument['column']]]
                h = pad(sequence=h, value=0, right=False).to(self.device)
                f = pad(sequence=f, value=0, right=False).to(self.device)
                batch[argument['column']] = h, f
                continue

            if(argument['column'] in ['b{}'.format(i) for i in range(1, 26)] + ['r2']):

                h = [h for h, _ in batch[argument['column']]]
                f = [f for _, f in batch[argument['column']]]
                h = pad(sequence=h, value=0, right=False).squeeze(2).to(self.device)
                f = pad(sequence=f, value=0, right=False).squeeze(2).to(self.device)
                batch[argument['column']] = h, f
                continue
            
            continue
        
        return(batch)

    pass

# target = 'b25'
# x = [v[v!=1] for v in batch[target].split(1, 1)]
# pad()