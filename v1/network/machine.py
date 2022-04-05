
import os
import tqdm
import torch
import numpy
import pickle
import pandas
import json
from sklearn import metrics

class history:

    def __init__(self):

        self.epoch = []
        self.loss   = {'train':[], "validation":[]}
        self.metric = {'train':[], "validation":[]}
        return

    pass

class metric:
    
    def __init__(self, limit):

        self.limit = limit
        return

    def compute(self, prediction, target):

        group = [prediction, target]
        score = []
        for prediction, target in zip(group[0], group[1]):

            top = min(self.limit, len(target))
            if(top<12): prediction = prediction[:top]
            if(top==12): target = target[:top]
            match = [1*(p==t) for p, t in zip(prediction, target)]
            precision = []
            for i, _ in enumerate(match):
                
                p = sum(match[:i+1]) if(match[i]==1) else 0
                precision += [p/(i+1)]
                pass

            score += [sum(precision) / top]
            pass

        score = numpy.mean(score)
        return(score)

    pass

class machine:

    def __init__(self, model=None, device='cpu', folder='log'):

        self.model      = model
        self.device     = device
        self.folder     = folder
        return

    def prepare(self):

        self.cost       = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
        self.history    = history()
        self.checkpoint = 0
        self.metric = metric(limit=12)
        os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
        return

    def learn(self, train=None, validation=None):

        self.history.epoch += [self.checkpoint]
        pass

        if(train):

            self.model = self.model.to(self.device)
            self.model.train()
            iteration  = {
                'total loss':[],
                'map@12 score':[]
            }
            progress = tqdm.tqdm(train, leave=False)
            for batch in progress:
                
                self.model.zero_grad()
                target = 'article_id_code'
                vector = ['FN', 'Active', 'age', 'club_member_status', 'fashion_news_frequency', 'postal_code']
                sequence = [
                    'price', 'sales_channel_id', 'product_code', 'prod_name', 'product_type_no', 
                    'product_type_name', 'product_group_name', 'graphical_appearance_no', 
                    'graphical_appearance_name', 'colour_group_code', 'colour_group_name', 
                    'perceived_colour_value_id', 'perceived_colour_value_name', 'perceived_colour_master_id', 
                    'perceived_colour_master_name', 'department_no', 'department_name', 'index_code', 
                    'index_name', 'index_group_no', 'index_group_name', 'section_no', 'section_name', 
                    'garment_group_no', 'garment_group_name', 'detail_desc', 'article_id_code'
                ]
                pass
                
                ##  Vector.
                for k in vector: 
                    
                    batch[k] = batch[k].to(self.device) 
                    pass

                ##  Sequence.
                for k in sequence: 
                    
                    batch[k]['history'] = batch[k]['history'].to(self.device) 
                    batch[k]['future'] = batch[k]['future'].to(self.device) 
                    pass

                likelihood, prediction = self.model(batch)
                pass

                loss = 0.0
                loss += self.cost(likelihood.flatten(0,1), batch[target]['future'][1:,].flatten(0,1))
                loss.backward()
                self.optimizer.step()
                pass

                ##  Metric.
                score = 0.0
                truth = [i.split() for i in batch['item']['article_id_code']]
                score += metric.compute(prediction, truth)
                pass

                iteration['total loss'] += [round(loss.item(), 3)]
                iteration['map@12 score'] += [round(score, 3)]
                pass

                value = (
                    iteration['total loss'][-1],
                    iteration['map@12 score'][-1]
                )
                message = "[train] total loss : {} | map@12 score : {}".format(*value)
                progress.set_description(message)
                pass
            
            self.history.loss['train'] += [round(numpy.array(iteration['total loss']).mean(), 3)]
            self.history.metric['train'] += [round(numpy.array(iteration['map@12 score']).mean(), 3)]
            pass

        if(validation):

            self.model = self.model.to(self.device)
            self.model.eval()
            iteration  = {
                'total loss':[],
                'map@12 score':[]
            }
            progress = tqdm.tqdm(train, leave=False)
            for batch in progress:
                
                with torch.no_grad():

                    target = 'article_id_code'
                    vector = ['FN', 'Active', 'age', 'club_member_status', 'fashion_news_frequency', 'postal_code']
                    sequence = [
                        'price', 'sales_channel_id', 'product_code', 'prod_name', 'product_type_no', 
                        'product_type_name', 'product_group_name', 'graphical_appearance_no', 
                        'graphical_appearance_name', 'colour_group_code', 'colour_group_name', 
                        'perceived_colour_value_id', 'perceived_colour_value_name', 'perceived_colour_master_id', 
                        'perceived_colour_master_name', 'department_no', 'department_name', 'index_code', 
                        'index_name', 'index_group_no', 'index_group_name', 'section_no', 'section_name', 
                        'garment_group_no', 'garment_group_name', 'detail_desc', 'article_id_code'
                    ]
                    pass
                    
                    ##  Vector.
                    for k in vector: 
                        
                        batch[k] = batch[k].to(self.device) 
                        pass

                    ##  Sequence.
                    for k in sequence: 
                        
                        batch[k]['history'] = batch[k]['history'].to(self.device) 
                        batch[k]['future'] = batch[k]['future'].to(self.device) 
                        pass

                    likelihood, prediction = self.model(batch)
                    pass

                    loss = 0.0
                    loss += self.cost(likelihood.flatten(0,1), batch[target]['future'][1:,].flatten(0,1))
                    pass

                ##  Metric.
                score = 0.0
                truth = [i.split() for i in batch['item']['article_id_code']]
                score += metric.compute(prediction, truth)
                pass

                iteration['total loss'] += [round(loss.item(), 3)]
                iteration['map@12 score'] += [round(score, 3)]
                pass

                value = (
                    iteration['total loss'][-1],
                    iteration['map@12 score'][-1]
                )
                message = "[train] total loss : {} | map@12 score : {}".format(*value)
                progress.set_description(message)
                pass
                        
            self.history.loss['validation'] += [round(numpy.array(iteration['total loss']).mean(), 3)]
            self.history.metric['validation'] += [round(numpy.array(iteration['map@12 score']).mean(), 3)]
            pass

        return

    def update(self, what='checkpoint'):

        if(what=='checkpoint'): self.checkpoint = self.checkpoint + 1
        return

    def save(self, what='history', mode='default'):

        if(what=='history'):

            with open(os.path.join(self.folder, 'loss'), 'w') as paper: 
                
                json.dump(self.history.loss, paper)
                pass

            pass

            with open(os.path.join(self.folder, 'metric'), 'w') as paper: 
                
                json.dump(self.history.metric, paper)
                pass

            pass

        if(what=='checkpoint'):

            if(mode=='default'):

                path = os.path.join(self.folder, "model-{}.checkpoint".format(self.checkpoint))
                with open(path, 'wb') as paper: 
                    
                    pickle.dump(self.model, paper)
                    pass

                pass
            
            if(mode=='better'):

                new = self.history.metric['validation'][-1]
                before = max([0.0] + self.history.metric['validation'][:-1])
                evolve = (new >= before)
                if(evolve):

                    print("new map@12 : {} >= before map@12 : {} | execute save model".format(new, before), end = '\n')
                    path = os.path.join(self.folder, "better-model-{}.checkpoint".format(new))
                    with open(path, 'wb') as paper: 
                        
                        pickle.dump(self.model, paper)
                        pass
                    
                    pass
                
                else:

                    print("new map@12 : {} < before map@12 : {} | skip save model".format(new, before), end = '\n')
                    pass

                pass

        return

    pass

