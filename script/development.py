
import library
import data
import network

##  Load all data table.
table = data.table(source='preprocess')
table.f1 = table.f1.loc[table.f1['seq_len']<100]
# table.f1 = table.f1.sample(200).reset_index(drop=True)
fold = 20
split = data.split(table=table.f1, method='fold', size=fold)

##  Each fold.
score = {'train':[], "validation":[]}
for k in split.iterate():

    print("start fold {}".format(k))
    split.get(fold=k)
    dataset = data.dataset(train=split.train, validation=split.validation)
    pass

    loader = data.loader(batch=32)
    loader.define(train=dataset.train, validation=dataset.validation)
    pass

    model = network.v5.model()
    machine = network.machine(model=model, device='cuda', folder='log/fold({})'.format(k))
    machine.prepare()
    pass

    epoch = 10
    for e in range(epoch):

        machine.learn(train=loader.train, validation=loader.validation)
        machine.save(what='history')
        machine.save(what='checkpoint', mode='better')
        machine.update(what='checkpoint')
        continue
    
    score['train'] += [max(machine.history.metric['train'])]
    score['validation'] += [max(machine.history.metric['validation'])]
    pass

value = library.numpy.mean(score['train']), library.numpy.mean(score['validation'])
print("[train] map@12 : {} | [validation] map@12 : {}".format(*value))
pass
