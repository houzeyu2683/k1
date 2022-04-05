
import numpy
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from functools import partial
import random

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
        h, f = 'history', 'future'
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            engine.prepare()
            pass

            key = 'edge'
            if(key in batch): batch[key] += [engine.edge]
            if(key not in batch): batch[key] = [engine.edge]
            pass

            key = 'item'
            if(key in batch): batch[key] += [pandas.DataFrame(engine.item).transpose()]
            if(key not in batch): batch[key] = [pandas.DataFrame(engine.item).transpose()]
            pass

            ##  Vector.
            vector = engine.handle(step="vector")
            pass

            key = ["FN", "Active", "age", "club_member_status", "fashion_news_frequency", "postal_code"]
            for k in key:

                if(k in batch): batch[k] += [vector[k]]
                if(k not in batch): batch[k] = [vector[k]]
                pass
            
            ##  Sequence
            sequence = engine.handle(step='sequence')
            pass

            key = [
                'price', 'article_code', 'sales_channel_id', 'product_code', 
                'prod_name', 'product_type_no', 'product_type_name', 
                'product_group_name', 'graphical_appearance_no', 
                'graphical_appearance_name', 'colour_group_code', 
                'colour_group_name', 'perceived_colour_value_id', 
                'perceived_colour_value_name', 'perceived_colour_master_id', 
                'perceived_colour_master_name', 'department_no', 
                'department_name', 'index_code', 'index_name', 
                'index_group_no', 'index_group_name', 'section_no', 
                'section_name', 'garment_group_no', 'garment_group_name', 
                'detail_desc', 't_dat_d', 'price_d', 'article_code_d', 
                'sales_channel_id_d', 'product_code_d', 'prod_name_d', 
                'product_type_no_d', 'product_type_name_d', 
                'product_group_name_d', 'graphical_appearance_no_d', 
                'graphical_appearance_name_d', 'colour_group_code_d', 
                'colour_group_name_d', 'perceived_colour_value_id_d', 
                'perceived_colour_value_name_d', 
                'perceived_colour_master_id_d', 
                'perceived_colour_master_name_d', 
                'department_no_d', 'department_name_d', 'index_code_d', 
                'index_name_d', 'index_group_no_d', 'index_group_name_d', 
                'section_no_d', 'section_name_d', 'garment_group_no_d', 
                'garment_group_name_d', 'detail_desc_d'
            ]
            for k in key:

                h, f = 'history', 'future'
                if(k in batch): batch[k][h] += [sequence[k][h]]
                if(k in batch): batch[k][f] += [sequence[k][f]]
                if(k not in batch): batch[k] = {h:[sequence[k][h]], f:[sequence[k][f]]}
                pass
            
            pass
        
        ##
        batch['item']  = pandas.concat(batch['item'])
        pass
        
        ##  Vector.
        batch['FN'] = torch.cat(batch["FN"], 0)
        batch['Active'] = torch.cat(batch["Active"], 0)
        batch['age'] = torch.cat(batch['age'], 0)
        batch['club_member_status'] = torch.cat(batch["club_member_status"], 1)
        batch["fashion_news_frequency"] = torch.cat(batch["fashion_news_frequency"], 1)
        batch["postal_code"] = torch.cat(batch["postal_code"], 1)
        pass
        
        ##  Sequence.
        key = [
            'price', 'article_code', 'sales_channel_id', 'product_code', 
            'prod_name', 'product_type_no', 'product_type_name', 
            'product_group_name', 'graphical_appearance_no', 
            'graphical_appearance_name', 'colour_group_code', 
            'colour_group_name', 'perceived_colour_value_id', 
            'perceived_colour_value_name', 'perceived_colour_master_id', 
            'perceived_colour_master_name', 'department_no', 
            'department_name', 'index_code', 'index_name', 
            'index_group_no', 'index_group_name', 'section_no', 
            'section_name', 'garment_group_no', 'garment_group_name', 
            'detail_desc', 't_dat_d', 'price_d', 'article_code_d', 
            'sales_channel_id_d', 'product_code_d', 'prod_name_d', 
            'product_type_no_d', 'product_type_name_d', 
            'product_group_name_d', 'graphical_appearance_no_d', 
            'graphical_appearance_name_d', 'colour_group_code_d', 
            'colour_group_name_d', 'perceived_colour_value_id_d', 
            'perceived_colour_value_name_d', 
            'perceived_colour_master_id_d', 
            'perceived_colour_master_name_d', 
            'department_no_d', 'department_name_d', 'index_code_d', 
            'index_name_d', 'index_group_no_d', 'index_group_name_d', 
            'section_no_d', 'section_name_d', 'garment_group_no_d', 
            'garment_group_name_d', 'detail_desc_d'
        ]
        h, f = 'history', 'future'
        for k in key:

            batch[k][h] = rnn.pad_sequence(batch[k][h], batch_first=False, padding_value=0).squeeze(-1)
            batch[k][f] = rnn.pad_sequence(batch[k][f], batch_first=False, padding_value=0).squeeze(-1)
            pass

        return(batch)

    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item.copy()
        self.mode = mode
        return
    
    def prepare(self):

        target = 'article_code'
        self.item['track_length'] = len(self.item[target].split())
        self.edge = random.randint(0, self.item['track_length']-1)
        return

    def handle(self, step=''):

        output = dict()

        if(step=="vector"):
            
            ##  基礎特徵工程.
            key = ["FN", "Active", "age"]
            for k in key:

                output[k] = torch.tensor(self.item[[k]]).type(torch.FloatTensor).unsqueeze(0)
                pass
            
            key = ["club_member_status", "fashion_news_frequency", "postal_code"]
            for k in key:

                output[k] = torch.tensor(self.item[[k]]).unsqueeze(0).type(torch.LongTensor)
                pass

            ##  延伸特徵工程.
            pass

        if(step=='sequence'):

            top = 12
            pass

            ##  基礎特徵工程.
            key = ['price']
            for k in key:

                period = [float(i) for i in self.item[k].split()]
                history, future = period[:self.edge], period[self.edge:][:top]
                history = [0.0] + history
                future  = [0.0] + future
                convert = lambda x: torch.tensor(x).unsqueeze(1).type(torch.FloatTensor)
                output[k] = {"history":convert(history), 'future':convert(future)}
                pass
            
            key = [
                'article_code', 'sales_channel_id',
                'product_code', 'prod_name', 'product_type_no',
                'product_type_name', 'product_group_name', 'graphical_appearance_no',
                'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
                'perceived_colour_value_id', 'perceived_colour_value_name',
                'perceived_colour_master_id', 'perceived_colour_master_name',
                'department_no', 'department_name', 'index_code', 'index_name',
                'index_group_no', 'index_group_name', 'section_no', 'section_name',
                'garment_group_no', 'garment_group_name', 'detail_desc'
            ]
            for k in key:

                period = [int(i) for i in self.item[k].split()]
                history, future = period[:self.edge], period[self.edge:][:top]
                history = [1.0] + history
                future  = [1.0] + future
                convert = lambda x: torch.tensor(x).unsqueeze(1).type(torch.LongTensor)
                output[k] = {"history":convert(history), 'future':convert(future)}
                pass

            pass

            ##  延伸特徵工程.
            key = ['t_dat', 'price']
            for k in key:

                period = [float(i) for i in self.item[k].split()]            
                history, future = period[:self.edge], period[self.edge:][:top]
                history, future = self.delta(history), self.delta(future)
                convert = lambda x: torch.tensor(x).unsqueeze(1).type(torch.FloatTensor)
                output[k+"_d"] = {"history":convert(history), 'future':convert(future)}
                pass
            
            key = [
                'article_code', 'sales_channel_id',
                'product_code', 'prod_name', 'product_type_no',
                'product_type_name', 'product_group_name', 'graphical_appearance_no',
                'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
                'perceived_colour_value_id', 'perceived_colour_value_name',
                'perceived_colour_master_id', 'perceived_colour_master_name',
                'department_no', 'department_name', 'index_code', 'index_name',
                'index_group_no', 'index_group_name', 'section_no', 'section_name',
                'garment_group_no', 'garment_group_name', 'detail_desc'
            ]
            for k in key:

                period = [int(i) for i in self.item[k].split()]            
                history, future = period[:self.edge], period[self.edge:][:top]
                history, future = self.delta(history), self.delta(future)
                history, future = [1*(h!=0) for h in history], [1*(f!=0) for f in future]
                convert = lambda x: torch.tensor(x).unsqueeze(1).type(torch.LongTensor)
                output[k+"_d"] = {"history":convert(history), 'future':convert(future)}
                pass

            pass

        return(output)

    def delta(self, x):
        
        if(x==[]): 
        
            x = [0]
            pass

        else:

            x = x[0:1] + x
            x = numpy.array(x[1:]) - numpy.array(x[:-1])
            x = [0] + x.tolist()
            pass
        
        y = x
        return(y)

    pass
