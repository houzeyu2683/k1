
import library
import data
import network

##  Load all data table.
table = data.table(source='preprocess')
split = data.split(table=table.f1, method='fold', size=10)

k = 1
split.get(fold=k)
dataset = data.dataset(train=split.train, validation=split.validation)
loader = data.loader(batch=64)
loader.define(train=dataset.train, validation=dataset.validation, test=None)

model = network.model()
machine = network.machine(model=model, device='cuda', folder='./cache')
machine.prepare()

for e in range(20):

    machine.learn(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    machine.save(what='checkpoint')
    machine.update(what='checkpoint')
    continue







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




