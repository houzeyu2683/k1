
import os
import tqdm
import torch
import numpy
import pickle
import pandas
import json
from sklearn import metrics

class history:

    epoch = []
    loss   = {'train':[]}#, "validation":[]}
    metric = {'train':[]}#, "validation":[]}
    pass

class machine:

    def __init__(self, model=None, device='cpu', folder='log'):

        self.model      = model
        self.device     = device
        self.folder     = folder
        return

    def prepare(self):

        self.cost       = {"ce":torch.nn.CrossEntropyLoss(), "mse" : torch.nn.MSELoss()}
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
        self.history    = history
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
                "sequence(price) loss":[],
                'sequence(article_code) loss':[],
                'total loss':[],
                'map@12 score':[]
            }
            progress = tqdm.tqdm(train, leave=False)
            for batch in progress:

                self.model.zero_grad()
                batch['row(numeric)'] = batch['row(numeric)'].to(self.device)
                batch['row(category)'] = batch['row(category)'].to(self.device)
                batch['sequence(price)']['history'] = batch['sequence(price)']['history'].to(self.device)
                batch['sequence(price)']['future'] = batch['sequence(price)']['future'].to(self.device)
                batch['sequence(article_code)']['history'] = batch['sequence(article_code)']['history'].to(self.device)
                batch['sequence(article_code)']['future'] = batch['sequence(article_code)']['future'].to(self.device)
                o = self.model(batch)
                point = min(len(batch['sequence(article_code)']['future']), 12)
                loss = dict()
                loss['sequence(price)'] = self.cost['mse'](
                    o['next(price)'][0:point,:,:], 
                    batch['sequence(price)']['future'][0:point,:,:]
                )
                loss['sequence(article_code)'] = self.cost['ce'](
                    o['next(article_code)'][0:point,:].flatten(0,1), 
                    batch['sequence(article_code)']['future'][0:point,:].flatten(0)
                )
                loss['total'] = sum(loss.values())
                loss['total'].backward()
                self.optimizer.step()
                pass
                
                ##  Loss value.
                iteration['sequence(price) loss'] += [round(loss['sequence(price)'].item(), 3)]
                iteration['sequence(article_code) loss'] += [round(loss['sequence(article_code)'].item(), 3)]
                iteration['total loss'] += [round(loss['total'].item(), 3)]
                pass

                ##  Metric value.
                point = min(len(batch['sequence(article_code)']['future']), 12)
                prediction = o['next(article_code)'][0:point,:,:].argmax(2).split(1, dim=1)
                target = batch['sequence(article_code)']['future'][0:point,:].split(1, dim=1)
                prediction = [i.squeeze().tolist() for i in prediction]
                target = [i.squeeze().tolist() for i in target]
                iteration['map@12 score'] += [self.metric.compute(prediction, target)]
                pass

                value = (
                    iteration['sequence(price) loss'][-1], 
                    iteration['sequence(article_code) loss'][-1], 
                    iteration['total loss'][-1],
                    iteration['map@12 score'][-1]
                )
                message = "[train] sequence(price) loss : {} | sequence(article_code) loss : {} | total loss : {} | map@12 score : {}".format(*value)
                progress.set_description(message)
                pass
            
            self.history.loss['train'] += [round(numpy.array(iteration['total loss']).mean(), 3)]
            self.history.metric['train'] += [round(numpy.array(iteration['map@12 score']).mean(), 3)]
            pass

        # if(validation):

        #     self.model = self.model.to(self.device)
        #     self.model.eval()
        #     pass
            
        #     iteration  = {'validation loss':[], 'validation accuracy':[]}
        #     progress = tqdm.tqdm(validation, leave=False)
        #     for batch in progress:

        #         with torch.no_grad():

        #             b = dict()
        #             b['i'] = batch['i'].to(self.device)
        #             b['ii'] = batch['ii'].to(self.device)
        #             b['iii'] = [(h.to(self.device), f.to(self.device)) for h, f in batch['iii']]
        #             x = [b[k] for k in b]
        #             o = self.model(x)
        #             loss = 0.0
        #             for l in range(len(o)):

        #                 if(l+1<len(o)): loss += self.cost[0](o[l], b['iii'][l][1][-1,:])
        #                 if(l+1==len(o)): loss += self.cost[1](o[l], b['iii'][l][1][-1,:])
        #                 pass
                    
        #             accuracy = metrics.accuracy_score(b['iii'][0][1][-1,:].cpu().numpy(), o[0].argmax(1).cpu().numpy())
        #             iteration['validation loss'] += [round(loss.item(), 3)]
        #             iteration['validation accuracy'] += [accuracy]
        #             progress.set_description("validation loss : {} | validation accuracy : {}".format(iteration['validation loss'][-1], iteration['validation accuracy'][-1]))
        #             pass
                
        #         continue

            # self.history.loss['validation'] += [round(numpy.array(iteration['validation loss']).mean(),3)]
            # self.history.accuracy['validation'] += [round(numpy.array(iteration['validation accuracy']).mean(),3)]
            # pass

        return

    def update(self, what='checkpoint'):

        if(what=='checkpoint'): self.checkpoint = self.checkpoint+1
        return

    def save(self, what='history'):

        if(what=='history'):

            with open(os.path.join(self.folder, 'loss'), 'w') as paper: json.dump(self.history.loss, paper)
            pass

            with open(os.path.join(self.folder, 'metric'), 'w') as paper: json.dump(self.history.accuracy, paper)
            pass

        if(what=='checkpoint'):

            path = os.path.join(self.folder, "model-{}.checkpoint".format(self.checkpoint))
            with open(path, 'wb') as paper: pickle.dump(self.model, paper)
            pass

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