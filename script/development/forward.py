
import library
import data
import network

##  Load all data table.
table = data.table(source='preprocess')
fold  = 20
split = data.split(table=table.f1, method='fold', size=fold)
split.get(fold=1)
dataset = data.dataset(train=split.train, validation=split.validation)
loader = data.loader(batch=4)
loader.define(train=dataset.train, validation=dataset.validation)
batch = next(iter(loader.train))


# x = [batch['item'][0].transpose(), batch['item'][1].transpose()]
# library.pandas.concat(objs=x)
# batch['article_code']['length']['history']

# vector = network.v1.vector()
# vector(batch).shape

# sequence = network.v1.sequence()
# sequence(batch)['article_code']['history'].shape
# sequence(batch)['article_code']['future'].shape

# fusion = network.v1.fusion()
# fusion(batch)

suggestion = network.v1.suggestion()
# suggestion(batch)

machine = network.v1.machine(model=suggestion, device='cpu')
machine.prepare()
machine.learn(train=loader.train)







