
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
embedding, likelihood, prediction, hit = machine.model(batch)
l = machine.cost[0](likelihood.flatten(0, 1), batch[target][f][1:,:].flatten(0, 1))
l = machine.cost[1](embedding['upgrade'].flatten(0,1), embedding['origin'].flatten(0,1), library.torch.cat(hit, 0))

##  Metric.
prediction = [i.tolist() for i in prediction]
truth = [i.squeeze(1).tolist() for i in batch[target][f][1:,:].split(1,1)]
truth = [list(filter((0).__ne__, i)) for i in truth]
machine.metric.compute(prediction, truth)

##  Epoch
machine.learn(train=loader.train)

