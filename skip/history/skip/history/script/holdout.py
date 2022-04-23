
##  Packages.
import data, process

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv", number=None)
table = data.tabulation.filter(table=table, column='mode', value='train')
train, check = data.validation.split(table, classification=None, ratio=0.1)

image = process.image()
target = process.target()

train['dataset'] = data.dataset(train['table'], image=image.learn, target=target.learn)
check['dataset'] = data.dataset(check['table'], image=image.guide, target=target.guide)
# case = train['dataset'].get(0)

##
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=32)
# batch = next(iter(loader.train))

##
import network


##
model = network.model()
# output = model.forward(batch)

criterion = network.criterion.cel()

##
optimizer = network.optimizer.adadelta(model)

##
folder   = "SOURCE/LOG"

##
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")
# machine.load(what='weight', path='SOURCE/LOG/113.checkpoint')
# machine.checkpoint = 114

##
iteration = 200
history = {
    'train' : {"loss":[]},
    'check' : {"loss":[]}
}
for epoch in range(iteration):

    if(epoch>=0):

        ##  Learning process.
        machine.learn(train=loader.train)

        if(epoch%1==0):

            machine.measure(train=loader.train, check=loader.check)
            machine.save("checkpoint")
            machine.save("measurement")

            ##  Measurement.
            measurement = machine.measurement
            
            ##  History of epoch.
            history['train']['loss'] += [measurement['train']['loss']]
            history['check']['loss'] += [measurement['check']['loss']]
            
            ##  Save the report.
            report = network.report(train=history['train'], check=history['check'])
            report.summarize()
            report.save(folder=folder)
            pass

        ##  Update.
        machine.update('schedule')
        machine.update('checkpoint')
        pass

    pass


# import torch
# cuda = torch.device('cuda')     # Default CUDA device
# cuda0 = torch.device('cuda:0')
# x = torch.tensor([1., 2.])

# a = torch.empty(5, 7, dtype=torch.cuda.DoubleTensor)
# a.to('cuda')