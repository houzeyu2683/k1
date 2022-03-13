
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

        self.cost       = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
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
                b['variable'] = batch['variable'].to(self.device)
                b['sequence'] = batch['sequence'].to(self.device)
                b['score']  = self.model(x = b['variable'], device=self.device)
                loss = self.cost(b["score"], b['sequence'][0,:])
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
                    b['variable'] = batch['variable'].to(self.device)
                    b['sequence']  = batch['sequence'].to(self.device)
                    b['score']  = self.model(x=b['variable'], device=self.device)
                    loss = self.cost(b["score"], b['sequence'][0,:])
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

    # def evaluate(self, loader):

    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass
        
    #     progress = tqdm.tqdm(loader, leave=False)
    #     iteration = {'image':[], "target":[], "prediction":[]}
    #     for index, batch in enumerate(progress, 0):
            
    #         with torch.no_grad():

    #             x = batch['image'].to(self.device)
    #             prediction = self.model.predict(
    #                 x=x, 
    #                 device=self.device, 
    #                 limit=50
    #             )
    #             target = batch['text'].squeeze().tolist()
    #             pass

    #             if(batch['item']['image'].item() not in iteration['image']): 
                    
    #                 iteration['image'] += [batch['item']['image'].item()]
    #                 iteration['target'] += [[target]]
    #                 iteration['prediction'] += [prediction]
    #                 pass
                
    #             else:

    #                 index = iteration['image'].index(batch['item']['image'].item())
    #                 iteration['target'][index] += [target]
    #                 pass
                
    #             pass

    #         pass
        
    #     bleu = []
    #     bleu += [round(corpus_bleu(iteration['target'], hypotheses=iteration['prediction'], weights=(1, 0, 0, 0)), 3)]
    #     bleu += [round(corpus_bleu(iteration['target'], hypotheses=iteration['prediction'], weights=(0.5, 0.5, 0, 0)), 3)]
    #     bleu += [round(corpus_bleu(iteration['target'], hypotheses=iteration['prediction'], weights=(0.33, 0.33, 0.33, 0)), 3)]
    #     bleu += [round(corpus_bleu(iteration['target'], hypotheses=iteration['prediction'], weights=(0.25, 0.25, 0.25, 0.25)), 3)]  
    #     return(bleu)

# s = corpus_bleu(list_of_references=[[[12,32,44,13], [12,32,22,13]], [[1,2,3,5], [5,4,3,2,1]]], hypotheses=[[12,32,44,13], [1,2,3,4,5]])
# round(s,3)


    # def evaluate(self, loader, name):

    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass
        
    #     iteration  = {'target':[], 'prediction':[]}
    #     result = {'image':[], "text":[], "description":[]}
    #     progress = tqdm.tqdm(loader, leave=False)
    #     for _, batch in enumerate(progress, 1):
            
    #         batch['item'] = batch['item'].reset_index(drop=True)
    #         for b in range(batch['size']):

    #             with torch.no_grad():
                    
    #                 v = dict()
    #                 v['x'] = batch['image'][b:b+1,:, :, :].to(self.device)
    #                 v['prediction'] = self.model.predict(
    #                     x=v['x'], 
    #                     device=self.device, 
    #                     limit=20
    #                 )
    #                 v['description'] = self.model.vocabulary.decode(token=v['prediction'], text=True)
    #                 v['target'] = [filter(
    #                     lambda x: (x!= self.model.vocabulary.index['<padding>']), 
    #                     batch['text'][:,b].tolist()
    #                 )]
    #                 pass
                
    #             iteration['target'] += [v['target']]
    #             iteration['prediction'] += [v['prediction']]
    #             pass

    #             result['image'] += [batch['item']['image'][b]]
    #             result['text'] +=  [batch['item']['text'][b]]
    #             result['description'] +=  [v['description']]
    #             pass

    #         pass
        
    #     score = round(corpus_bleu(iteration['target'], hypotheses=iteration['prediction']),3)
    #     print('the gleu score {}'.format(score))
    #     pass

    #     location = os.path.join(self.folder, "{} result.csv".format(name))
    #     pandas.DataFrame(result).to_csv(location, index=False)
    #     return

    # def save(self, what='checkpoint'):

    #     ##  Save the checkpoint.
    #     if(what=='checkpoint'):

    #         path = os.path.join(self.folder, "model-" + str(self.checkpoint) + ".checkpoint")
    #         with open(path, 'wb') as data:

    #             pickle.dump(self.model, data)
    #             pass

    #         print("save the weight of model to {}".format(path))
    #         pass

    #     if(what=='history'):

    #         history = pandas.DataFrame(self.track)
    #         history.to_csv(
    #             os.path.join(self.folder, "history.csv"), 
    #             index=False
    #         )
    #         pass

    #         # path = os.path.join(self.folder, "history.html")
    #         figure = go.Figure()
    #         figure.add_trace(
    #             go.Scatter(
    #                 x=history['epoch'], y=history['train loss'],
    #                 mode='lines+markers',
    #                 name='train loss',
    #                 line_color="blue"
    #             )
    #         )
    #         figure.add_trace(
    #             go.Scatter(
    #                 x=history['epoch'], y=history['validation loss'],
    #                 mode='lines+markers',
    #                 name='validation loss',
    #                 line_color="green"
    #             )
    #         )
    #         figure.write_html(os.path.join(self.folder, "history.html"))
    #         pass

    #     return

    # def update(self, what='checkpoint'):

    #     if(what=='checkpoint'): self.checkpoint = self.checkpoint + 1
    #     return

    # def load(self, path, what='model'):

    #     if(what=='model'):

    #         with open(path, 'rb') as paper:
                
    #             self.model = pickle.load(paper)

    #         print('load model successfully')
    #         pass

    #     return
        
    # pass






    # def evaluate(self, train=None, validation=None):

    #     ################################################################################
    #     ##  On train.
    #     ################################################################################
    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass
        
    #     record  = {'train edit distance' : []}
    #     progress = tqdm.tqdm(train, leave=False)
    #     for batch in progress:

    #         with torch.no_grad():
                
    #             value = {}
    #             value['image'] = batch['image'].to(self.device)
    #             value['text']  = batch['text'].to(self.device)
    #             for i in range(batch['size']):

    #                 prediction = self.model.autoregressive(image=batch['image'][i:i+1,:,:,:], device=self.device, limit=20)
    #                 prediction = self.model.vocabulary.reverse(x=prediction)
    #                 target = batch['item'].iloc[i]["text"]
    #                 record['train edit distance'] += [editdistance.eval(prediction, target)]
    #                 pass

    #             pass
            
    #         pass

    #     self.history['train edit distance'] += [numpy.array(record['train edit distance']).mean()]
    #     pass
    
    #     ################################################################################
    #     ##  On validation.
    #     ################################################################################
    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass
        
    #     record  = {'validation edit distance' : []}
    #     progress = tqdm.tqdm(validation, leave=False)
    #     for batch in progress:

    #         with torch.no_grad():
                
    #             value = {}
    #             value['image'] = batch['image'].to(self.device)
    #             value['text']  = batch['text'].to(self.device)
    #             for i in range(batch['size']):

    #                 prediction = self.model.autoregressive(image=batch['image'][i:i+1,:,:,:], device=self.device, limit=20)
    #                 prediction = self.model.vocabulary.reverse(x=prediction)
    #                 target = batch['item'].iloc[i]["text"]
    #                 record['validation edit distance'] += [editdistance.eval(prediction, target)]
    #                 pass

    #             pass
            
    #         pass

    #     self.history['validation edit distance'] += [numpy.array(record['validation edit distance']).mean()]
    #     return