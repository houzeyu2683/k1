

##
##  Packages.
import os, tqdm, torch, numpy, pickle, pandas


##
##  Class for machine learning process, case by case.
class machine:

    def __init__(self, model, optimizer=None, criterion=None, device='cuda', folder=None, checkpoint="0"):

        self.model      = model
        self.optimizer  = optimizer
        self.criterion  = criterion
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass

        ##  Create the folder for storage.
        if(self.folder):
        
            os.makedirs(self.folder, exist_ok=True)
            pass
        
        ##  Optimizer schedule.
        if(self.optimizer):

            self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9, last_epoch=-1, verbose=False)
            pass
    
    def bundle(self, batch):

        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        return(image, target)

    def learn(self, loader):

        ##  Mode of learn.
        self.model.train()
        self.model = self.model.to(self.device)
        pass

        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            ##  Handle batch.
            batch = self.bundle(batch)

            ##  Update weight.
            self.optimizer.zero_grad()
            score, target = self.model(batch)
            loss = self.criterion.to(self.device)(score, target)
            loss.backward()
            self.optimizer.step()

            ##  Show message to progress bar.
            loss = round(loss.item(), 3)
            progress.set_description("loss:{}".format(loss))
            pass
        
        pass

    def evaluate(self, loader):

        self.model.eval()
        self.model = self.model.to(self.device)
        pass

        evaluation = {
            "loss" : []
        }
        for batch in tqdm.tqdm(loader, leave=False):

            with torch.no_grad():
            
                batch = self.bundle(batch)
                score, target = self.model(batch)
                loss  = self.criterion(score, target).cpu().detach().numpy().item()
                evaluation['loss'] += [loss]
                pass
            
            pass

        ##  Summarize record.
        evaluation['loss']  = numpy.mean(evaluation['loss'])
        return(evaluation)

    def save(self, what='checkpoint'):

        ##  Save the checkpoint.
        if(what=='checkpoint'):

            path = os.path.join(self.folder, self.checkpoint+".checkpoint")
            torch.save(self.model.state_dict(), path)
            print("save the weight of model to {}".format(path))
            pass

        ##  Save the measurement.
        if(what=='measurement'):    

            path = os.path.join(self.folder, self.checkpoint + ".measurement")
            with open(path, 'wb') as paper:

                pickle.dump(self.measurement, paper)
                pass

            print("save the checkpoint of measurement to {}".format(path))
            pass
  
    def update(self, what='checkpoint'):

        if(what=='checkpoint'):
            
            try:
                
                self.checkpoint = str(int(self.checkpoint) + 1)
                print("update the checkpoint to {} for next iteration\n".format(self.checkpoint))
                pass

            except:

                print("the checkpoint is not integer, skip update checkpoint\n")
                pass

        if(what=='schedule'):

            self.schedule.step()
            rate = self.optimizer.param_groups[0]['lr']
            print("learning rate update to {}".format(rate))
            pass

    def load(self, what='weight', path=None):

        if(what=='weight'):
            
            try:

                self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print("finish loading")
                return
            
            except:

                print("fail loading")
                return        
            
            pass

        return
    
    def predict(self, loader):

        self.model.eval()
        self.model = self.model.to(self.device)
        pass

        likelihood = []
        for batch in tqdm.tqdm(loader, leave=False):

            with torch.no_grad():
            
                batch = self.bundle(batch)
                score, target = self.model(batch)
                pass
            
            likelihood += [score.cpu().numpy()]
            pass

        ##  Summarize record.
        likelihood  = numpy.concatenate(likelihood, axis=0)
        likelihood  = pandas.DataFrame(likelihood, columns=["c"+str(i) for i in range(10)])
        return(likelihood)

    pass

        # event = {'train':train, "check":check}
        # for key in event:

        #     if(event[key]):

        #         record = {
        #             'loss':[]
        #         }
        #         for batch in tqdm.tqdm(event[key], leave=False):

        #             with torch.no_grad():

        #                 batch = self.bundle(batch)
        #                 score, target = self.model(batch)
        #                 loss  = self.criterion(score, target).cpu().detach().numpy().item()
        #                 pass

        #             record['loss']  += [loss]
        #             pass
                
        #         ##  Summarize record.
        #         record['loss']  = numpy.mean(record['loss'])
        #         pass

        #         ##  Insert evaluation to measurement.
        #         measurement[key] = record
        #         print("end of measure the {}".format(key))
        #         pass

        #     pass

        # self.measurement = measurement
        # pass

    # def predict(self, test):

    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass

    #     likelihood = []
    #     # prediction = []
    #     for batch in tqdm.tqdm(test, leave=False):

    #         image, target = batch
    #         batch = image.to(self.device), target.to(self.device)
    #         score = self.model(batch).cpu().detach().numpy()
    #         likelihood += [score] 
    #         # prediction += score.argmax(dim=1)
    #         pass
        
    #     likelihood = pandas.DataFrame(numpy.concatenate(likelihood, axis=0), columns=["c"+ str(i) for i in range(10)])
    #     # prediction = numpy.array(prediction)
    #     return(likelihood)




# from itertools import chain
# x = [2]
# z= chain(*x)


# import torch
# import torch.nn as nn
# loss = nn.CrossEntropyLoss()

# x = torch.randn((2,6,3))
# y = torch.randint(0,3,(2,6))
# x.shape
# y.shape
# loss(x[0,:,:], y[0,:])

# x[0,:,:]
# output.shape
# target = target.to('cuda')
# torch.flatten(target).shape
# loss   = criterion.to('cuda')(output, torch.flatten(target))