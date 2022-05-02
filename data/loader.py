
##  The packages
import os
import numpy
import random
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from functools import partial

class argument:

    reservation = {
        "customer_id":{"column":"customer_id", "type":str},
        "length":{"column":"length", "type":int}
    }
    personality = {
        'p1':{"column":"FN", "type":torch.FloatTensor}, 
        'p2':{"column":'Active', "type":torch.FloatTensor}, 
        'p3':{"column":"club_member_status", "type":torch.LongTensor},
        'p4':{"column":'fashion_news_frequency', "type":torch.LongTensor}, 
        'p5':{"column":'age', "type":torch.FloatTensor}, 
        'p6':{"column":'postal_code', "type":torch.LongTensor}
    }
    behavior = {
        'b1':{"column":'product_code', "type":torch.LongTensor}, 
        'b2':{"column":'prod_name', "type":torch.LongTensor}, 
        'b3':{"column":'product_type_no', "type":torch.LongTensor},
        'b4':{"column":'product_type_name', "type":torch.LongTensor}, 
        'b5':{"column":'product_group_name', "type":torch.LongTensor}, 
        'b6':{"column":'graphical_appearance_no', "type":torch.LongTensor},
        'b7':{"column":"graphical_appearance_name", "type":torch.LongTensor}, 
        'b8':{"column":"colour_group_code", "type":torch.LongTensor},
        'b9':{"column":"colour_group_name", "type":torch.LongTensor},
        'b10':{"column":'perceived_colour_value_id', "type":torch.LongTensor}, 
        'b11':{"column":"perceived_colour_value_name", "type":torch.LongTensor},
        'b12':{"column":"perceived_colour_master_id", "type":torch.LongTensor}, 
        'b13':{"column":"perceived_colour_master_name", "type":torch.LongTensor},
        'b14':{"column":'department_no', "type":torch.LongTensor}, 
        'b15':{"column":"department_name", "type":torch.LongTensor}, 
        'b16':{"column":"index_code", "type":torch.LongTensor}, 
        'b17':{"column":"index_name", "type":torch.LongTensor},
        'b18':{"column":"index_group_no", "type":torch.LongTensor}, 
        'b19':{"column":"index_group_name", "type":torch.LongTensor}, 
        'b20':{"column":"section_no", "type":torch.LongTensor}, 
        'b21':{"column":"section_name", "type":torch.LongTensor},
        'b22':{"column":"garment_group_no", "type":torch.LongTensor}, 
        'b23':{"column":"garment_group_name", "type":torch.LongTensor}, 
        'b24':{"column":"detail_desc", "type":torch.LongTensor},
        'b25':{"column":"article_label", "type":torch.LongTensor},
        'r1':{"column":'price', "type":torch.FloatTensor},
        'r2':{"column":'sales_channel_id', "type":torch.LongTensor}
    }
    pass

class process:
    
    def __init__(self, item=None, mode='train'):
    
        self.item = item.copy()
        self.mode = mode
        return

    def handle(self):

        output = dict()
        pass

        if(self.mode in ['train', 'validation']):

            for claim, style in argument.reservation.items():

                x = self.item[[claim]]
                output[claim] = x.astype(style['type']).item()
                continue

            for claim, style in argument.personality.items():
    
                # x = self.item[[claim]]
                # output[claim] = x.astype(style['type']).item()

                x = [float(self.item[claim])]
                x = torch.tensor([x])
                output[claim] = x.type(style['type'])
                continue

            for claim, style in argument.behavior.items():

                x = [float(i) for i in self.item[claim].split()]
                x = [1.0] + x
                x = torch.cat([torch.tensor([[i]]) for i in x], dim=0)
                x = x.type(style['type'])
                h, f = x[:-1], x[1:]
                output[claim] = h, f        
                continue

            pass

        return(output)

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

    def pad(self, sequence='list of tensor', value=0, right=True):
        
        if(right): 
            
            tensor = rnn.pad_sequence(sequence, batch_first=False, padding_value=value)
            pass

        else:

            sequence = [i.flip(dims=[0]) for i in sequence]
            sequence = rnn.pad_sequence(sequence, batch_first=False, padding_value=value)
            tensor = sequence.flip(dims=[0])        
            pass

        return(tensor)

    def collect(self, iteration, mode):
        
        batch = dict()
        batch.update({'size':len(iteration)})
        batch.update({'mode':mode})
        batch.update({'item':pandas.concat(iteration, axis=1).transpose()})
        pass
    
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            dictionary = engine.handle()
            for key, value in dictionary.items(): 
                
                batch[key] = batch[key] + [value] if(key in batch) else [value]
                pass

            pass

        for claim, _ in argument.personality.items():

            if(claim in ['p1', 'p2', 'p5']):
                
                x = torch.cat(batch[claim], dim=0)
                batch[claim] = x.to(self.device)
                pass

            if(claim in ['p3', 'p4', 'p6']):
                
                x = torch.cat(batch[claim], dim=0).squeeze(1)
                batch[claim] = x.to(self.device)
                pass

            pass

        for claim, _ in argument.behavior.items():
    
            h = [h for h, _ in batch[claim]]
            f = [f for _, f in batch[claim]]
            pass

            if(claim in ['b{}'.format(i) for i in range(1, 26)] + ['r2']):
    
                h = self.pad(sequence=h, value=0, right=False).squeeze(2).to(self.device)
                f = self.pad(sequence=f, value=0, right=False).squeeze(2).to(self.device)
                pass

            if(claim in ['r1']):
    
                h = self.pad(sequence=h, value=0, right=False).to(self.device)
                f = self.pad(sequence=f, value=0, right=False).to(self.device)
                pass

            batch[claim] = h, f
            pass
        
        return(batch)

    pass



        # for _, argument in schedule.items():

        #     if(argument['column'] in ['p1', 'p2', 'p5']):

        #         x = torch.cat(batch[argument['column']], dim=0)
        #         batch[argument['column']] = x.to(self.device)
        #         continue

        #     if(argument['column'] in ['p3', 'p4', 'p6']):

        #         x = torch.cat(batch[argument['column']], dim=0).squeeze(1)
        #         batch[argument['column']] = x.to(self.device)
        #         continue

        #     if(argument['column'] in ['r1']):
                
        #         h = [h for h, _ in batch[argument['column']]]
        #         f = [f for _, f in batch[argument['column']]]
        #         h = pad(sequence=h, value=0, right=False).to(self.device)
        #         f = pad(sequence=f, value=0, right=False).to(self.device)
        #         batch[argument['column']] = h, f
        #         continue

        #     if(argument['column'] in ['b{}'.format(i) for i in range(1, 26)] + ['r2']):

        #         h = [h for h, _ in batch[argument['column']]]
        #         f = [f for _, f in batch[argument['column']]]
        #         h = pad(sequence=h, value=0, right=False).squeeze(2).to(self.device)
        #         f = pad(sequence=f, value=0, right=False).squeeze(2).to(self.device)
        #         batch[argument['column']] = h, f
        #         continue
            
            # continue

# target = 'b25'
# x = [v[v!=1] for v in batch[target].split(1, 1)]
# pad()





            # for origin, argument in schedule.items():

            #     if(origin in ['customer_id', 'length']): 
                    
            #         x = self.item[[argument['column']]]
            #         output[argument['column']] = x.astype(argument['type']).item()
            #         continue
                
                # if('p' in argument['column']):

                #     x = torch.tensor([[float(self.item[argument['column']])]])
                #     x = x.type(argument['type'])
                #     output[argument['column']] = x
                #     continue

                # if('b' in argument['column']):
    
                #     x = [float(i) for i in self.item[argument['column']].split()]
                #     x = [1.0] + x
                #     x = torch.cat([torch.tensor([[i]]) for i in x], dim=0)
                #     x = x.type(argument['type'])
                #     h, f = x[:-1], x[1:]
                #     output[argument['column']] = h, f
                #     continue

                # if('r' in argument['column']):
        
                #     x = [float(i) for i in self.item[argument['column']].split()]
                #     x = [1.0] + x
                #     x = torch.cat([torch.tensor([[i]]) for i in x], dim=0)
                #     x = x.type(argument['type'])
                #     h, f = x[:-1], x[1:]
                #     output[argument['column']] = h, f
                #     continue

                # continue
            


# schedule = {
    # 'customer_id':{'column':'customer_id', "type":str}, 
    # 'length':{"column":'length', "type":int}, 
    # 'FN':{"column":"p1", "type":torch.FloatTensor}, 
    # 'Active':{"column":'p2', "type":torch.FloatTensor}, 
    # 'club_member_status':{"column":"p3", "type":torch.LongTensor},
    # 'fashion_news_frequency':{"column":'p4', "type":torch.LongTensor}, 
    # 'age':{"column":'p5', "type":torch.FloatTensor}, 
    # 'postal_code':{"column":'p6', "type":torch.LongTensor}, 
    # 'product_code':{"column":'b1', "type":torch.LongTensor}, 
    # 'prod_name':{"column":'b2', "type":torch.LongTensor}, 
    # 'product_type_no':{"column":'b3', "type":torch.LongTensor},
    # 'product_type_name':{"column":'b4', "type":torch.LongTensor}, 
    # 'product_group_name':{"column":'b5', "type":torch.LongTensor}, 
    # 'graphical_appearance_no':{"column":'b6', "type":torch.LongTensor},
    # 'graphical_appearance_name':{"column":"b7", "type":torch.LongTensor}, 
    # 'colour_group_code':{"column":"b8", "type":torch.LongTensor},
    # 'colour_group_name':{"column":"b9", "type":torch.LongTensor},
    # 'perceived_colour_value_id':{"column":'b10', "type":torch.LongTensor}, 
    # 'perceived_colour_value_name':{"column":"b11", "type":torch.LongTensor},
    # 'perceived_colour_master_id':{"column":"b12", "type":torch.LongTensor}, 
    # 'perceived_colour_master_name':{"column":"b13", "type":torch.LongTensor},
    # 'department_no':{"column":'b14', "type":torch.LongTensor}, 
    # 'department_name':{"column":"b15", "type":torch.LongTensor}, 
    # 'index_code':{"column":"b16", "type":torch.LongTensor}, 
    # 'index_name':{"column":"b17", "type":torch.LongTensor},
    # 'index_group_no':{"column":"b18", "type":torch.LongTensor}, 
    # 'index_group_name':{"column":"b19", "type":torch.LongTensor}, 
    # 'section_no':{"column":"b20", "type":torch.LongTensor}, 
    # 'section_name':{"column":"b21", "type":torch.LongTensor},
    # 'garment_group_no':{"column":"b22", "type":torch.LongTensor}, 
    # 'garment_group_name':{"column":"b23", "type":torch.LongTensor}, 
    # 'detail_desc':{"column":"b24", "type":torch.LongTensor},
    # 'article_label':{"column":"b25", "type":torch.LongTensor},
    # 'price':{"column":'r1', "type":torch.FloatTensor},
    # 'sales_channel_id':{"column":'r2', "type":torch.LongTensor} 
# }