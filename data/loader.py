
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
        h, f = 'history', 'future'
        for item in iteration:
            
            engine = process(item=item, mode=mode)
            engine.prepare()
            pass

            batch['edge'] += [engine.edge]
            batch['item'] += [pandas.DataFrame(engine.item).transpose()]
            name = []
            pass

            vector = engine.handle(step="vector")
            pass

            name += ["FN", "Active", "age"] + ["club_member_status", "fashion_news_frequency", "postal_code"]
            if(name[0] in batch): batch[name[0]] += [vector[name[0]]]
            if(name[0] not in batch): batch[name[0]] = [vector[name[0]]]
            if(name[1] in batch): batch[name[1]] += [vector[name[1]]]
            if(name[1] not in batch): batch[name[1]] = [vector[name[1]]]
            if(name[2] in batch): batch[name[2]] += [vector[name[2]]]
            if(name[2] not in batch): batch[name[2]] = [vector[name[2]]]
            if(name[3] in batch): batch[name[3]] += [vector[name[3]]]
            if(name[3] not in batch): batch[name[3]] = [vector[name[3]]]
            if(name[4] in batch): batch[name[4]] += [vector[name[4]]]
            if(name[4] not in batch): batch[name[4]] = [vector[name[4]]]
            if(name[5] in batch): batch[name[5]] += [vector[name[5]]]
            if(name[5] not in batch): batch[name[5]] = [vector[name[5]]]
            pass

            sequence = engine.handle(step='sequence')
            pass

            name += ["article_code"]
            if(name[6] in batch): batch[name[6]][h] += [sequence[name[6]][h]]
            if(name[6] in batch): batch[name[6]][f] += [sequence[name[6]][f]]
            if(name[6] not in batch): batch[name[6]] = {h:[sequence[name[6]][h]], f:[sequence[name[6]][f]]}
            pass
            
            name += ["price"]
            if(name[7] in batch): batch[name[7]][h] += [sequence[name[7]][h]]
            if(name[7] in batch): batch[name[7]][f] += [sequence[name[7]][f]]
            if(name[7] not in batch): batch[name[7]] = {h:[sequence[name[7]][h]], f:[sequence[name[7]][f]]}
            pass
        
        batch['item']  = pandas.concat(batch['item'])
        # batch['truth'] = [i.split() for i in batch['item']['article_code']]
        batch[name[0]] = torch.cat(batch[name[0]], 0)
        batch[name[1]] = torch.cat(batch[name[1]], 0)
        batch[name[2]] = torch.cat(batch[name[2]], 0)
        batch[name[3]] = torch.cat(batch[name[3]], 1)
        batch[name[4]] = torch.cat(batch[name[4]], 1)
        batch[name[5]] = torch.cat(batch[name[5]], 1)
        pass

        batch[name[6]][h] = rnn.pad_sequence(batch[name[6]][h], batch_first=False, padding_value=0).squeeze(-1)
        batch[name[6]][f] = rnn.pad_sequence(batch[name[6]][f], batch_first=False, padding_value=0).squeeze(-1)
        batch[name[7]][h] = rnn.pad_sequence(batch[name[7]][h], batch_first=False, padding_value=0).squeeze(-1)
        batch[name[7]][f] = rnn.pad_sequence(batch[name[7]][f], batch_first=False, padding_value=0).squeeze(-1)
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
            
            output = dict()
            name = ["FN", "Active", "age"] + ["club_member_status", "fashion_news_frequency", "postal_code"]
            pass

            output[name[0]] = torch.tensor(self.item[[name[0]]]).type(torch.FloatTensor).unsqueeze(0)
            output[name[1]] = torch.tensor(self.item[[name[1]]]).type(torch.FloatTensor).unsqueeze(0)
            output[name[2]] = torch.tensor(self.item[[name[2]]]).type(torch.FloatTensor).unsqueeze(0)
            pass
            
            output[name[3]] = torch.tensor(self.item[[name[3]]]).unsqueeze(0).type(torch.LongTensor)
            output[name[4]] = torch.tensor(self.item[[name[4]]]).unsqueeze(0).type(torch.LongTensor)
            output[name[5]] = torch.tensor(self.item[[name[5]]]).unsqueeze(0).type(torch.LongTensor)
            pass

        if(step=='sequence'):

            output = dict()
            name = ['article_code', 'price']
            h, f = 'history', 'future'
            pass

            line = [int(i) for i in self.item[name[0]].split()]
            history, future = [1.0] + line[:self.edge], [2.0] + line[self.edge:][:12]
            history = torch.tensor(history).unsqueeze(1).type(torch.LongTensor)
            future  = torch.tensor(future).unsqueeze(1).type(torch.LongTensor)
            output[name[0]] = {h:history, f:future}
            pass

            line = [float(i) for i in self.item[name[1]].split()]
            history, future = [1.0] + line[:self.edge], [2.0] + line[self.edge:][:12]
            history = torch.tensor(history).unsqueeze(1).type(torch.FloatTensor)
            future  = torch.tensor(future).unsqueeze(1).type(torch.FloatTensor)
            output[name[1]] = {h:history, f:future}
            pass

        return(output)

    pass
