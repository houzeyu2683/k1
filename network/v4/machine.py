
import os
import tqdm
import torch
import numpy
import pickle
import pandas
import json

class history:

    epoch = []
    loss = {'train':[], "validation":[]}
    accuracy = {'train':[], "validation":[]}
    pass

class machine:

    def __init__(self, model=None, device='cpu', folder='log'):

        self.model      = model
        self.device     = device
        self.folder     = folder
        return

    def prepare(self):

        self.cost       = [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()]
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
        self.history    = history
        self.checkpoint = 0
        os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
        return

    def learn(self, train=None, validation=None):

        self.history.epoch += [self.checkpoint]
        pass

        if(train):

            self.model = self.model.to(self.device)
            self.model.train()
            iteration  = {'train loss':[]}
            progress = tqdm.tqdm(train, leave=False)
            for batch in progress:

                self.model.zero_grad()
                b = dict()
                b['i'] = batch['i'].to(self.device)
                b['ii'] = batch['ii'].to(self.device)
                b['iii'] = [(h.to(self.device), f.to(self.device)) for h, f in batch['iii']]
                x = [b[k] for k in b]
                o = self.model(x)
                loss = 0.0
                for l in range(len(o)):

                    if(l+1<len(o)): loss += self.cost[0](o[l], b['iii'][l][1][-1,:])
                    if(l+1==len(o)): loss += self.cost[1](o[l], b['iii'][l][1][-1,:])
                    pass
                
                # loss = self.cost(o[0], o[1], o[2])
                loss.backward()
                self.optimizer.step()
                iteration['train loss'] += [loss.item()]
                progress.set_description("train loss : {}".format(round(iteration['train loss'][-1],3)))
                pass
            
            self.history.loss['train'] += [round(numpy.array(iteration['train loss']).mean(), 3)]
            pass

        if(validation):

            self.model = self.model.to(self.device)
            self.model.eval()
            pass
            
            iteration  = {'validation loss':[]}
            progress = tqdm.tqdm(validation, leave=False)
            for batch in progress:

                with torch.no_grad():

                    b = dict()
                    b['i'] = batch['i'].to(self.device)
                    b['ii'] = batch['ii'].to(self.device)
                    b['iii'] = [(h.to(self.device), f.to(self.device)) for h, f in batch['iii']]
                    x = [b[k] for k in b]
                    o = self.model(x)
                    loss = 0.0
                    for l in range(len(o)):

                        if(l+1<len(o)): loss += self.cost[0](o[l], b['iii'][l][1][-1,:])
                        if(l+1==len(o)): loss += self.cost[1](o[l], b['iii'][l][1][-1,:])
                        pass
    
                    iteration['validation loss'] += [loss.item()]
                    progress.set_description("validation loss : {}".format(round(iteration['validation loss'][-1], 3)))
                    pass
                
                continue

            self.history.loss['validation'] += [round(numpy.array(iteration['validation loss']).mean(),3)]
            pass

        return

    def update(self, what='checkpoint'):

        if(what=='checkpoint'): self.checkpoint = self.checkpoint+1
        return

    def save(self, what='history'):

        if(what=='history'):

            with open(os.path.join(self.folder, 'loss'), 'w') as paper: json.dump(self.history.loss, paper)
            pass

        if(what=='checkpoint'):

            path = os.path.join(self.folder, "model-{}.checkpoint".format(self.checkpoint))
            with open(path, 'wb') as paper: pickle.dump(self.model, paper)
            pass

        return

    pass
