
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
        self.loss   = {'train':[], 'validation':[]}
        return

    pass

class machine:

    def __init__(self, model=None, device='cpu', folder='log'):

        self.model      = model
        self.device     = device
        self.folder     = folder
        return

    def prepare(self):

        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
        self.history    = history()
        self.checkpoint = 0
        os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
        return

    def learn(self, train=None, validation=None):

        self.history.epoch += [self.checkpoint]
        pass

        if(train):

            self.model = self.model.to(self.device)
            self.model.train()
            iteration  = {
                'total loss':[]
            }
            progress = tqdm.tqdm(train, leave=False)
            for batch in progress:
                
                self.model.zero_grad()
                loss = self.model.cost(batch)
                loss['total'].backward()
                self.optimizer.step()
                pass

                iteration['total loss'] += [round(loss['total'].item(), 3)]
                pass

                value = (
                    iteration['total loss'][-1],
                )
                message = "[train] total loss : {}".format(*value)
                progress.set_description(message)
                pass
            
            self.history.loss['train'] += [round(numpy.array(iteration['total loss']).mean(), 3)]
            pass
        
        if(validation):
    
            self.model = self.model.to(self.device)
            self.model.eval()
            iteration  = {
                'total loss':[]
            }
            progress = tqdm.tqdm(validation, leave=False)
            for batch in progress:
                
                with torch.no_grad(): loss = self.model.cost(batch)
                iteration['total loss'] += [round(loss['total'].item(), 3)]
                pass

                value = (
                    iteration['total loss'][-1],
                )
                message = "[validation] total loss : {}".format(*value)
                progress.set_description(message)
                pass
            
            self.history.loss['validation'] += [round(numpy.array(iteration['total loss']).mean(), 3)]
            pass
        
        return
        # if(validation):

        #     self.model = self.model.to(self.device)
        #     self.model.eval()
        #     iteration  = {
        #         'total loss':[],
        #         'map@12 score':[]
        #     }
        #     progress = tqdm.tqdm(train, leave=False)
        #     for batch in progress:
                
        #         with torch.no_grad():

        #             target = "product_code"#'article_id_code'
        #             vector = ['FN', 'Active', 'age', 'club_member_status', 'fashion_news_frequency', 'postal_code']
        #             sequence = [
        #                 'price', 'sales_channel_id', 'product_code', 'prod_name', 'product_type_no', 
        #                 'product_type_name', 'product_group_name', 'graphical_appearance_no', 
        #                 'graphical_appearance_name', 'colour_group_code', 'colour_group_name', 
        #                 'perceived_colour_value_id', 'perceived_colour_value_name', 'perceived_colour_master_id', 
        #                 'perceived_colour_master_name', 'department_no', 'department_name', 'index_code', 
        #                 'index_name', 'index_group_no', 'index_group_name', 'section_no', 'section_name', 
        #                 'garment_group_no', 'garment_group_name', 'detail_desc', 'article_id_code'
        #             ]
        #             pass
                    
        #             ##  Vector.
        #             for k in vector: 
                        
        #                 batch[k] = batch[k].to(self.device) 
        #                 pass

        #             ##  Sequence.
        #             for k in sequence: 
                        
        #                 batch[k]['history'] = batch[k]['history'].to(self.device) 
        #                 batch[k]['future'] = batch[k]['future'].to(self.device) 
        #                 pass

        #             likelihood, prediction = self.model(batch)
        #             pass

        #             loss = 0.0
        #             loss += self.cost(likelihood.flatten(0,1), batch[target]['future'][1:,].flatten(0,1))
        #             pass

        #         ##  Metric.
        #         score = 0.0
        #         truth = [i.split() for i in batch['item'][target]]
        #         score += self.metric.compute(prediction, truth)
        #         pass

        #         iteration['total loss'] += [round(loss.item(), 3)]
        #         iteration['map@12 score'] += [round(score, 3)]
        #         pass

        #         value = (
        #             iteration['total loss'][-1],
        #             iteration['map@12 score'][-1]
        #         )
        #         message = "[train] total loss : {} | map@12 score : {}".format(*value)
        #         progress.set_description(message)
        #         pass
                        
        #     self.history.loss['validation'] += [round(numpy.array(iteration['total loss']).mean(), 3)]
        #     self.history.metric['validation'] += [round(numpy.array(iteration['map@12 score']).mean(), 3)]
        #     pass

        # return

    # def save(self, what='history', mode='default'):

    #     if(what=='history'):

    #         with open(os.path.join(self.folder, 'loss'), 'w') as paper: 
                
    #             json.dump(self.history.loss, paper)
    #             pass

    #         pass

    #     if(what=='checkpoint'):

    #         if(mode=='default'):

    #             path = os.path.join(self.folder, "model-{}.checkpoint".format(self.checkpoint))
    #             with open(path, 'wb') as paper: 
                    
    #                 pickle.dump(self.model, paper)
    #                 pass

    #             pass
            
    #         if(mode=='better'):

    #             new = self.history.metric['validation'][-1]
    #             before = max([0.0] + self.history.metric['validation'][:-1])
    #             evolve = (new >= before)
    #             if(evolve):

    #                 print("new map@12 : {} >= before map@12 : {} | execute save model".format(new, before), end = '\n')
    #                 path = os.path.join(self.folder, "better-model-{}.checkpoint".format(new))
    #                 with open(path, 'wb') as paper: 
                        
    #                     pickle.dump(self.model, paper)
    #                     pass
                    
    #                 pass
                
    #             else:

    #                 print("new map@12 : {} < before map@12 : {} | skip save model".format(new, before), end = '\n')
    #                 pass

    #             pass

    #     return

    def save(self, what='history'):
    
        if(what=='history'):

            path = os.path.join(self.folder, 'loss')
            with open(path, 'w') as paper: json.dump(self.history.loss, paper)
            pass

        if(what=='checkpoint'):

            path = os.path.join(self.folder, "model-{}.checkpoint".format(self.checkpoint))
            with open(path, 'wb') as paper: pickle.dump(self.model, paper)
            pass

        return

    def update(self, what='checkpoint'):
    
        if(what=='checkpoint'): self.checkpoint = self.checkpoint + 1
        return

    pass

