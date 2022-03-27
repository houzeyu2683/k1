
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
        batch['edge'] = []
        batch['item'] = []
        batch["numeric"]                = []
        batch['club_member_status']     = []
        batch['fashion_news_frequency'] = []
        batch['postal_code']            = []
        batch["article_code"]           = {'history':[], 'future':[]}
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            engine.prepare()
            batch['edge'] += [engine.edge]
            batch['item'] += [engine.item]
            numeric, category = engine.handle(step="vector")
            batch["numeric"]                += [numeric]
            batch['club_member_status']     += [category[0]]
            batch['fashion_news_frequency'] += [category[1]]
            batch['postal_code']            += [category[2]]
            h, f = engine.handle(step='article_code')
            batch["article_code"]['history'] += [h]
            batch["article_code"]['future'] += [f]
            pass
        
        batch['numeric'] = torch.cat(batch['numeric'], 0)
        batch['club_member_status'] = torch.cat(batch['club_member_status'], 1)
        batch['fashion_news_frequency'] = torch.cat(batch['fashion_news_frequency'], 1)
        batch['postal_code'] = torch.cat(batch['postal_code'], 1)
        pass

        s, h, f, l = 'article_code', 'history', 'future', 'length'
        batch[s][h] = rnn.pad_sequence(batch[s][h], batch_first=False, padding_value=0).squeeze(-1)
        batch[s][f] = rnn.pad_sequence(batch[s][f], batch_first=False, padding_value=0).squeeze(-1)
        batch[s][l] = {h:len(batch[s][h]), f:len(batch[s][f])}
        return(batch)

    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        return
    
    def prepare(self):

        self.edge = random.randint(0, self.item['trans_length']-1)
        return

    def handle(self, step=''):

        ##  Handle vector numeric.
        if(step=="vector"):
            
            '''
            [item]
            customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
            FN                                                                    0.0
            Active                                                                0.0
            age                                                                  0.25
            club_member_status                                                      0
            fashion_news_frequency                                                  2
            postal_code                                                         57312
            '''
            output = []
            v = self.item[["FN", "Active", "age"]]
            v = torch.tensor(v).type(torch.FloatTensor).unsqueeze(0)
            output += [v]
            pass

            v = self.item[["club_member_status", "fashion_news_frequency", "postal_code"]]
            v = torch.tensor(v).unsqueeze(1).type(torch.LongTensor)
            v = v.split(1,0)
            output += [v]
            pass

        if(step=='article_code'):

            '''
            customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
            article_code                                          1 33994 19336 33751
            return ((length,1),(length,1))
            '''
            line = [int(i) for i in self.item['article_code'].split()]
            h, f = line[:self.edge], [2.0] + line[self.edge:][:12]
            if(h==[]): h += [1.0]
            history = torch.tensor(h).unsqueeze(1).type(torch.LongTensor)
            future  = torch.tensor(f).unsqueeze(1).type(torch.LongTensor)
            output = [history, future]            
            pass

        return(output)

    pass
