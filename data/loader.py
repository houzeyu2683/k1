
import numpy
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from functools import partial
import random

class constant:

    version = '6.0.0'
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
        batch["vector(numeric)"]                = []
        batch['vector(club_member_status)']     = []
        batch['vector(fashion_news_frequency)'] = []
        batch['vector(postal_code)']            = []
        batch["sequence(price)"]         = {'history':[], 'future':[]}
        batch["sequence(article_code)"]  = {'history':[], 'future':[]}
        # batch['sequence(t_dat_delta)']        = {'history':[], 'future':[]}
        # batch['sequence(article_code_delta)'] = {'history':[], 'future':[]}
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            engine.prepare()
            batch['edge'] += [engine.edge]
            batch['item'] += [engine.item]
            batch["vector(numeric)"] += [engine.handle(step="vector(numeric)")]
            c = engine.handle(step="vector(category)")
            batch['vector(club_member_status)'] += [c[0]]
            batch['vector(fashion_news_frequency)'] += [c[1]]
            batch['vector(postal_code)'] += [c[2]]
            h, f = engine.handle(step='sequence(price)')
            batch["sequence(price)"]['history'] += [h]
            batch["sequence(price)"]['future'] += [f]
            h, f = engine.handle(step='sequence(article_code)')
            batch["sequence(article_code)"]['history'] += [h]
            batch["sequence(article_code)"]['future'] += [f]
            # h, f = engine.handle(step='sequence(t_dat_delta)')
            # batch["sequence(t_dat_delta)"]['history'] += [h]
            # batch["sequence(t_dat_delta)"]['future'] += [f]
            # h, f = engine.handle(step='sequence(article_code_delta)')
            # batch["sequence(article_code_delta)"]['history'] += [h]
            # batch["sequence(article_code_delta)"]['future'] += [f]
            pass
        
        batch['vector(numeric)'] = torch.cat(batch['vector(numeric)'], 0)
        batch['vector(club_member_status)'] = torch.cat(batch['vector(club_member_status)'], 1)
        batch['vector(fashion_news_frequency)'] = torch.cat(batch['vector(fashion_news_frequency)'], 1)
        batch['vector(postal_code)'] = torch.cat(batch['vector(postal_code)'], 1)
        pass

        h = 'history'
        f = 'future'
        s = 'sequence(price)'
        batch[s][h] = rnn.pad_sequence(batch[s][h], batch_first=False, padding_value=0)
        batch[s][f] = rnn.pad_sequence(batch[s][f], batch_first=False, padding_value=0)
        s = 'sequence(article_code)'
        batch[s][h] = rnn.pad_sequence(batch[s][h], batch_first=False, padding_value=0).squeeze(-1)
        batch[s][f] = rnn.pad_sequence(batch[s][f], batch_first=False, padding_value=0).squeeze(-1)
        # s = 'sequence(t_dat_delta)'
        # batch[s][h] = rnn.pad_sequence(batch[s][h], batch_first=False, padding_value=0).squeeze(-1)
        # batch[s][f] = rnn.pad_sequence(batch[s][f], batch_first=False, padding_value=0).squeeze(-1)
        # s = 'sequence(article_code_delta)'
        # batch[s][h] = rnn.pad_sequence(batch[s][h], batch_first=False, padding_value=0).squeeze(-1)
        # batch[s][f] = rnn.pad_sequence(batch[s][f], batch_first=False, padding_value=0).squeeze(-1)
        return(batch)

    pass

class process:

    def __init__(self, item=None, mode=None):

        self.item = item
        self.mode = mode
        return
    
    def prepare(self):

        self.edge = random.randint(1, self.item['seq_len']-1)
        return

    def handle(self, step=''):

        ##  Handle vector numeric.
        if(step=="vector(numeric)"):
            
            '''
            customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
            FN                                                                    0.0
            Active                                                                0.0
            age                                                                  0.25
            return (1,3)
            '''
            v = self.item[["FN", "Active", "age"]]
            output = torch.tensor(v).type(torch.FloatTensor).unsqueeze(0)
            pass

        ##  Handle vector category.
        if(step=="vector(category)"):

            '''
            customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
            club_member_status                                                      0
            fashion_news_frequency                                                  2
            postal_code                                                         57312
            return (4,1)
            '''
            v = self.item[["club_member_status", "fashion_news_frequency", "postal_code"]]
            v = torch.tensor(v).unsqueeze(1).type(torch.LongTensor)
            output = v.split(1,0)
            pass

        if(step=='sequence(price)'):
            
            '''
            customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
            price                   0.0 1.1145845272206303 1.0429512893982806 1.05...
            return ((length,1),(length,1))
            '''
            line = [float(i) for i in self.item['price'].split()]
            h = line[:self.edge]
            f = line[self.edge:][:12]
            history = torch.tensor(h).unsqueeze(1).type(torch.FloatTensor)
            future  = torch.tensor(f).unsqueeze(1).type(torch.FloatTensor)
            output = [history, future]
            pass

        if(step=='sequence(article_code)'):

            '''
            customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
            article_code                                          1 33994 19336 33751
            return ((length,1),(length,1))
            '''
            line = [int(i) for i in self.item['article_code'].split()]
            h = line[:self.edge]
            f = line[self.edge:][:12] 
            history = torch.tensor(h).unsqueeze(1).type(torch.LongTensor)
            future  = torch.tensor(f).unsqueeze(1).type(torch.LongTensor)
            output = [history, future]            
            pass

        # if(step=='sequence(article_code_delta)'):

        #     '''
        #     customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
        #     article_code                                          1 33994 19336 33751
        #     return ((length,1),(length,1))
        #     '''
        #     line = [int(i) for i in self.item['article_code'].split()]
        #     line = [-1+2] + [2+(1*(start==end)) for start, end in zip(line[:-1], line[1:])]
        #     h = line[:self.edge]
        #     f = h[-1:] + line[self.edge:][:12]
        #     history = torch.tensor(h).unsqueeze(1).type(torch.LongTensor)
        #     future  = torch.tensor(f).unsqueeze(1).type(torch.LongTensor)
        #     output = [history, future]          
        #     pass
        
        # if(step=='sequence(t_dat_delta)'):

        #     '''
        #     customer_id             0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...
        #     t_dat_delta                                                       1 1 1 1
        #     return ((length,1),(length,1))
        #     '''
        #     line = [int(i) for i in self.item['t_dat_delta'].split()]
        #     h = line[:self.edge]
        #     f = line[self.edge:][:12]
        #     h = numpy.array(h[:1] + h) 
        #     f = numpy.array(h[-1:] + f)
        #     h = h[1:] - h[:-1]
        #     f = f[1:] - f[:-1]
        #     history = torch.tensor(h.tolist()).unsqueeze(1).type(torch.LongTensor)
        #     future  = torch.tensor(f.tolist()).unsqueeze(1).type(torch.LongTensor)
        #     output = [history, future]            
        #     pass

        return(output)

    pass
