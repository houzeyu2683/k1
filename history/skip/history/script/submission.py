
##  Packages.
import data

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv", number=None)
table = data.tabulation.filter(table=table, column='mode', value='test')

image = data.process.image()
target = data.process.target()

dataset = data.dataset(table, image=image.guide, target=target.guide)

##
loader = data.loader(test=dataset, batch=32)

##
import network

##
model = network.model()
criterion = network.criterion.cel()
optimizer = network.optimizer.adam(model)

machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=None, checkpoint="0")
machine.load(what='weight', path='SOURCE/LOG/113.checkpoint')
likelihood = machine.predict(test=loader.test)
table = table.join(likelihood)
table = table[['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']]
table.to_csv("SOURCE/LOG/SUBMISSION.csv", index=False)


# import torch
# x = table[['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']]
# import numpy 

# x = torch.nn.Softmax(dim=1)(torch.tensor(numpy.array(x)))
# import pandas
# table[['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']] = pandas.DataFrame(x.numpy())

# machine.checkpoint = 114

# ##
# iteration = 120
# history = {
#     'train' : {"cost":[]},
#     'check' : {"cost":[]}
# }
# for epoch in range(iteration):

#     if(epoch>=114):

#         ##  Learning process.
#         machine.learn(loader.train)

#         if(epoch%1==0):

#             machine.measure(train=loader.train, check=loader.check)
#             machine.save("checkpoint")
#             machine.save("measurement")

#             ##  Measurement.
#             measurement = machine.measurement
            
#             ##  History of epoch.
#             history['train']['cost'] += [measurement['train']['cost']]
#             history['check']['cost'] += [measurement['check']['cost']]
            
#             ##  Save the report.
#             report = network.report(train=history['train'], check=history['check'])
#             report.summarize()
#             report.save(folder=folder)
#             pass

#         ##  Update.
#         machine.update('schedule')
#         machine.update('checkpoint')
#         pass



