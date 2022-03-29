
import library
import data
import network

##  Load all data table.
table = data.table(source='preprocess')
fold  = 20
split = data.split(table=table.f1, method='fold', size=fold)
split.get(fold=1)
dataset = data.dataset(train=split.train, validation=split.validation)
loader = data.loader(batch=2)
loader.define(train=dataset.train, validation=dataset.validation)
batch = next(iter(loader.train))

##
target = 'article_code'
f = 'future'

##  Forward.
y = network.v1.vector()(batch)
y = network.v1.sequence()(batch)
y = network.v1.fusion()(batch)
y = network.v1.suggestion()(batch)

##  Loss.
machine = network.v1.machine(model=network.v1.suggestion())
machine.prepare()
likelihood, prediction, positive, negative = machine.model(batch)
l = machine.cost[0](likelihood.flatten(0,1), prediction.flatten(0,1))
l = machine.cost[1](positive[0].flatten(0,1), positive[1].flatten(0,1), positive[2].flatten(0,1))
l = machine.cost[1](negative[0].flatten(0,1), negative[1].flatten(0,1), negative[2].flatten(0,1))

##  Metric.
prediction = [i.squeeze(-1).tolist() for i in prediction.split(1,1)]
truth = [list(filter((0).__ne__, i)) for i in [i.squeeze(-1).tolist() for i in batch[target][f][1:,:].split(1,1)]]
machine.metric.compute(prediction, truth)

##  Epoch
machine.learn(train=loader.train)

