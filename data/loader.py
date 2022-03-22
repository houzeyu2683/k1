
from matplotlib.collections import Collection
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from functools import partial
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
        
        c = dict()
        c['size'] = len(iteration)
        c['mode'] = mode
        c['item'] = []
        c["row(numeric)"]    = []
        c["row(category)"]   = []
        c["sequence(price)"]  = {"history":[],"future":[]}
        c["sequence(article_code)"]  = {"history":[],"future":[]}
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            engine.prepare()
            c['item'] += [engine.item]
            c["row(numeric)"] += [engine.handle(step="row(numeric)")]
            c["row(category)"] += [engine.handle(step="row(category)")]
            h, f = engine.handle(step='sequence(price)')
            c["sequence(price)"]['history'] += [h]
            c["sequence(price)"]['future'] += [f]
            h, f = engine.handle(step='sequence(article_code)')
            c["sequence(article_code)"]['history'] += [h]
            c["sequence(article_code)"]['future'] += [f]
            pass

        c['row(numeric)'] = torch.cat(c['row(numeric)'], 0)
        c['row(category)'] = torch.cat(c['row(category)'], 1)
        c['sequence(price)']['history'] = rnn.pad_sequence(c['sequence(price)']['history'], batch_first=False, padding_value=0)
        c['sequence(price)']['future'] = rnn.pad_sequence(c['sequence(price)']['future'], batch_first=False, padding_value=0)
        c['sequence(article_code)']['history'] = rnn.pad_sequence(c['sequence(article_code)']['history'], batch_first=False, padding_value=0).squeeze(-1)
        c['sequence(article_code)']['future'] = rnn.pad_sequence(c['sequence(article_code)']['future'], batch_first=False, padding_value=0).squeeze(-1)
        # print(c['row(numeric)'].shape)
        # print(c['row(category)'].shape)
        # print(c['sequence(price)']['history'].shape)
        # print(c['sequence(price)']['future'].shape)
        # print(c['sequence(article_code)']['history'].shape)
        # print(c['sequence(article_code)']['future'].shape)
        collection = c
        return(collection)
    
    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        pass
    
    def prepare(self):

        limit = min([12, self.item['seq_len']])
        self.reservation = random.randint(1, limit)
        self.point = random.randint(0, self.item['seq_len'] - self.reservation)
        return

    ##  Handle item.
    def handle(self, step=''):

        '''
        >>> item
        customer_id                     00007d2de826758b65a93dd24ce629ed66842531df6699...
        FN                                                                            1.0
        Active                                                                        1.0
        club_member_status                                                              0
        fashion_news_frequency                                                          4
        age                                                                          0.32
        postal_code                                                                194979
        article_code                                          6390 46307 46308 46305 6389
        product_code                                          1743 18961 18961 18961 1743
        prod_name                                           19763 44534 44534 44534 19765
        product_type_no                                                    50 50 50 50 50
        product_type_name                                             103 103 103 103 103
        product_group_name                                                 10 10 10 10 10
        graphical_appearance_no                                            13 13 13 13 13
        graphical_appearance_name                                          18 18 18 18 18
        colour_group_code                                                  35 35 52 11 10
        colour_group_name                                                   43 43 9 10 20
        perceived_colour_value_id                                              5 10 7 7 5
        perceived_colour_value_name                                             8 7 4 4 8
        perceived_colour_master_id                                            7 7 21 15 3
        perceived_colour_master_name                                         15 15 8 9 18
        department_no                                                    206 90 90 90 206
        department_name                                                  208 44 44 44 208
        index_code                                                              6 3 3 3 6
        index_name                                                              7 9 9 9 7
        index_group_no                                                          4 3 3 3 4
        index_group_name                                                        4 5 5 5 4
        section_no                                                         44 10 10 10 44
        section_name                                                       18 47 47 47 18
        garment_group_no                                                     5 22 22 22 5
        garment_group_name                                                 10 17 17 17 10
        detail_desc                                         15803 41840 41840 41840 15803
        price                           1.0257593123209168 1.028624641833811 1.0286246...
        sales_channel_id                                                        4 4 4 4 4
        seq_len                                                                         5
        Name: 2, dtype: object
        '''

        ##  Handle row numeric.
        if(step=="row(numeric)"):
            
            v = self.item[["FN", "Active", "age"]]
            output = torch.tensor(v).type(torch.FloatTensor).unsqueeze(0)
            pass

        ##  Handle row category.
        if(step=="row(category)"):

            v = self.item[["club_member_status", "fashion_news_frequency", "postal_code"]]
            output = torch.tensor(v).type(torch.LongTensor).unsqueeze(1)
            pass

        if(step=='sequence(price)'):

            ##  Split history(x) and future(y) after handle sequence of numeric, 'price'.
            entirety = [float(v) for v in self.item['price'].split()]
            history, future = entirety[:self.point], entirety[self.point:][:self.reservation]
            if(history==[]): history += [0.0]
            history = torch.tensor(history).type(torch.FloatTensor).unsqueeze(1)
            future = torch.tensor(future).type(torch.FloatTensor).unsqueeze(1)
            output = history, future
            pass

        if(step=='sequence(article_code)'):

            ##  Split history(x) and future(y) after handle sequence of category, 'article_code'.
            entirety = [int(v) for v in self.item['article_code'].split()]
            history, future = entirety[:self.point], entirety[self.point:][:self.reservation]
            if(history==[]): history += [1]
            history = torch.tensor(history).type(torch.LongTensor).unsqueeze(1)
            future = torch.tensor(future).type(torch.LongTensor).unsqueeze(1)
            output = history, future
            pass

        
        return(output)
