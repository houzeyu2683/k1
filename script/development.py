
import library
import data
import network

##  Load all data table.
table = data.table(source='preprocess(sample)')
split = data.split(table=table.f1, method='fold', size=10)

##  Each fold.
k = 1
split.get(fold=k)
dataset = data.dataset(train=split.train, validation=split.validation)

loader = data.loader(batch=7)
loader.define(train=dataset.train, validation=dataset.validation, test=None)
batch = next(iter(loader.train))
model = network.v5.model()
machine = network.machine(model=model, device='cpu', folder='log')
machine.prepare()
machine.learn(train=loader.train)




metric = network.v5.metric(limit=12)


o = model(batch)


m = metric.compute(prediction, target)


m.evaluate()


m = metric(, [i.squeeze().tolist() for i in y.split(1, dim=1)])
m.evaluate()




len()


import torch.nn as nn
nn.MSELoss()(y, y_hat)

y = batch['sequence(article_code)']['future'][0:point,:]
y_hat = o['next(article_code)'][0:point,:,:]
y = y.flatten(0)
y_hat = y_hat.flatten(0,1)
nn.CrossEntropyLoss(ignore_index=0)(y_hat, y)

y.shape
y_hat.shape

o['next(article_code)'].shape


o['next(price)'].shape
o['next(article_code)'].shape

o['day(1)']['next(price)']
import torch
torch.cat([batch['sequence(price)']['history'], o['next(price)'].unsqueeze(0)], 0)

oo = o['next(article_code)'].argmax(1)
oo.unsqueeze(0)

# batch.keys()
# batch['row(numeric)']
# batch['row(numeric)'].shape
# batch['row(category)']
# batch['row(category)'].shape
# batch['sequence(price)']['history']
# batch['sequence(price)']['history'].shape
# batch['sequence(price)']['future']
# batch['sequence(price)']['future'].shape
# batch['sequence(article_code)']['history']
# batch['sequence(article_code)']['history'].shape
# batch['sequence(article_code)']['future']
# batch['sequence(article_code)']['future'].shape

# import torch
# torch.tensor(b['sequence(article_code)']['history']).shape
# torch.tensor(b['sequence(article_code)']['future']).shape
# # b['row numeric'].shape
# # b['row category'].shape
# # b['sequence(price)']['history'].shape
# b['sequence(price)']['future'].shape
# b['sequence(article_code)']['history'].shape
# b['sequence(article_code)']['future'].shape
# x = [1.0372206303724927]
# torch.tensor(x).unsqueeze(0)
# import torch
# from torch import nn
# torch.cat(b['row numeric'], 0).shape
# torch.cat(b['row category'], 1).shape

# from torch.nn.utils import rnn
# rnn.pad_sequence(b['sequence(price)']['history'], batch_first=False, padding_value=0)
# rnn.pad_sequence(b['sequence(article_code)']['history'], batch_first=False, padding_value=0).squeeze()
# rnn.pad_sequence(b['sequence(price)']['future'], batch_first=False, padding_value=0).shape
# rnn.pad_sequence(b['sequence(article_code)']['future'], batch_first=False, padding_value=0).squeeze()

# x = [b['i'], b['ii'], b['iii']]

network.v5.row()(batch)
network.v5.sequence()(batch)
network.v5.suggestion()(batch)
model = network.v5.model()
# o = model(x)
# from torch import nn
# import sklearn




# import torch.nn as nn
# l1 = nn.CrossEntropyLoss()
# l2 = nn.MSELoss()
# l1(o[0], b['iii'][0][1][-1,:])
# l2(o[-1], b['iii'][-1][1][-1,:])

# len(y)
# y[-1]


# len(b['iii'])

machine = network.v4.machine(model=model, device='cpu', folder='./log(v3)')
machine.prepare()

for e in range(50):

    machine.learn(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    machine.save(what='checkpoint')
    machine.update(what='checkpoint')
    continue

# for b in loader.train:

#     model([b['x1'], b['x2'], b['x3'], b['x4'], b['x5'], b['y']])
#     pass
# model.layer['x1'](b['x1'])
# model.layer['x2'](b['x2'])
# x = b['x2']

# table.f1.columns

# import torch
# v = torch.randn((3, 5))
# e = torch.randn((3, 5))
# t = torch.tensor([1,1,1])

# b = 3
# for r in range(b):
#     r = 0
#     v[r:r+1,:].repeat(b, 1), e, 
# v[0:1, :]
# e[-0:]

# machine.history.loss['train']

# batch  = next(iter(loader.train))
# batch['variable'][:,-1]
# x = batch['variable']
# x

# x1 = network.v1.variable()(x)
# x2 = network.v1.article()(x1)




# batch['sequence']
# for i in loader.train: print(i['sequence'])
    
# x = dataset.train.__getitem__(2)['sequence']
# x

# split.clear()

# vocabulary["663713001"]


# [i for i in split.train['truth'].head()]

# generation(split.train).__getitem__(1)

# # generation.train.shape
# # generation.train.head(3)
# # generation.validation.shape
# # generation.validation.head(3)
# # a = set(generation.validation['customer_id'])
# # b = set(generation.validation['customer_id'])
# # set.intersection(a,b)
# # train, test = generation.block[0]
# # table.user.loc[generation.block[0]['train']]
# # table.user.tail()

# # target = library.pandas.read_csv("resource/preprocess/csv/target.csv")
# # user = library.pandas.read_csv("resource/preprocess/csv/user.csv")

# # user['postal_code'].nunique()
# # user['postal_code'].min()

# # user['club_member_status'] = user['club_member_status'] / 4
# # user['fashion_news_frequency'] = user['fashion_news_frequency'] / 5
# # user['age'] = user['age'] / 100

# # user.to_csv('user.csv', index=False)

# # merge = library.pandas.concat([user, target], axis=1)

# # merge = merge.loc[:,~merge.columns.duplicated()]
# # merge.columns

# # merge.to_csv("f1.csv", index=False)




# b = next(iter(loader.train))
# b['y'].shape
# b['x3'].shape
# x1 = network.v3.x1()(b['x1'])
# x2 = network.v3.x2()(b['x2'])
# x3 = network.v3.x3()(b['x3'])
# x4 = network.v3.x4()(b['x4'])
# x5 = network.v3.x5()(b['x5'])
# m = network.v3.model()
# o = m([b['x1'],b['x2'],b['x3'],b['x4'],b['x5'],b['y']])
# # x = [b['x1'],b['x2'],b['x3'],b['x4'],b['x5'],b['y']]
# # o[0].shape
# # o[1].shape

# # b['y'][:,2:3].shape
# l = library.torch.nn.CosineEmbeddingLoss()
# l(o[0], o[1], o[2])
# # import torch
# # torch.stack([x1,x2], 0).shape
# # item = dataset.train.__getitem__(2)
# # b['x3'][:,0,:].shape
# # b = next(iter(loader.train))
# # m = network.model()
# # o = m([b['row'], b['sequence'], b['target']])
# # cost[0](o[0], b['target']) + cost[1](o[1], o[2], 1*(b['target']>-1))
# b['x3'][:,:,0]
