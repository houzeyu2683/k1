
import library
import data
import network

##  Load all data table.
table = data.table(source='preprocess(sample)')
fold  = 20
split = data.split(table=table.f1, method='fold', size=fold)
split.get(fold=1)
dataset = data.dataset(train=split.train, validation=split.validation)
loader = data.loader(batch=7)
loader.define(train=dataset.train, validation=dataset.validation)

##
b = next(iter(loader.train))
model = network.v5.vector()
y = model(b)
y.shape
model = network.v5.sequence()
y = model(b)
y.shape
model = network.v5.suggestion()
y = model(b)
y.shape

machine = network.v5.machine(model=model)
machine.prepare()
machine.learn(train=loader.train)

b.keys()

model = dict()


import torch
torch.tensor([1,2,3]).unsqueeze(0).split(1,1)[0]

