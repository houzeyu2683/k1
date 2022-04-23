
import v1

table = v1.data.table(source='resource/preprocess/csv')
table.load(file='group(train).csv')
table.file.head()

split = v1.data.split(table=table.file, group=None, method='fold', size=20)
split.get(fold=0)
split.train
split.validation

dataset = v1.data.dataset(train=split.train, validation=split.validation)
loader = v1.data.loader(batch=4)
loader.define(train=dataset.train, validation=dataset.validation)
next(iter(loader.train))

model = v1.network.model()
machine = v1.network.machine(model=model, device='cpu', folder=None)
machine.prepare()
# machine.metric.compute(prediction=[[0, 'b', 'c', 'c', "c"], ['a']], target=[['0', 'b', 'c'], ['k', 'o']])
machine.learn(train=loader.train)


