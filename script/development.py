
import library
import data
import network

##  Load all data table.
table = data.table(source='preprocess')
split = data.split(table=table.f1, method='fold', size=10)
# table.f1['article_code'].apply(lambda x: len(x.split())).max()
k = 1
split.get(fold=k)
dataset = data.dataset(train=split.train, validation=split.validation)
loader = data.loader(batch=36)
loader.define(train=dataset.train, validation=dataset.validation, test=None)

model = network.v3.model()
machine = network.v3.machine(model=model, device='cuda', folder='./cache')
machine.prepare()

for e in range(20):

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
