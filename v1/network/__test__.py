
import data

table = data.table(source="resource/preprocess/sample/")
table.load(file="csv/feature(train).csv")
table.file.shape
table.file.head(4)

split = data.split(table=table.file, group=None, method='fold', size=10)
split.get(fold=0)

dataset = data.dataset(train=split.train, validation=split.validation, test=None)

loader = data.loader(batch=16, device='cuda')
loader.define(train=dataset.train, validation=dataset.validation)

import v1
model = v1.network.model(device='cuda')
machine = v1.network.machine(model=model, device='cuda', folder='log')
machine.prepare()
for epoch in range(20):

    machine.learn(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    machine.update(what='checkpoint')
    pass

