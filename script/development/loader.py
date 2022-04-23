
import library
import v1
# data
# import network

##  Load all data table.
table = v1.data.table(source="resource/preprocess/csv/")
table.load(file="group(train).csv")
table.file.head()
pass

fold  = 20
split = v1.data.split(table=table.file, method='fold', size=fold)
split.get(fold=1)
dataset = v1.data.dataset(train=split.train, validation=split.validation)
pass

item = dataset.train.__getitem__(27)
engine = v1.data.process(item, 'train')
engine.prepare()
engine.handle(step='vector')
engine.handle(step='sequence')
pass

loader = v1.data.loader(batch=4)
loader.define(train=dataset.train)
batch = next(iter(loader.train))
batch.keys()
batch['price']['history'].shape
batch['club_member_status'].shape
batch['age'].shape







# ##  batch
# batch.keys()
# batch['FN'].shape
# batch['club_member_status'].shape
# batch['article_code']['history'].shape
# batch['article_code']['future'].shape
# batch['price']['history'].shape
# batch['price']['future'].shape

# v = network.v1.vector()(batch)
# v.shape

# sequence = network.v1.sequence()
# s = sequence(batch)
# s.keys()
# s['article_code']['history'].shape
# s['price']['history'].shape


# import torch 
# torch.cat(batch['FN'], 0)

# ##



# model = network.v5.vector()
# y = model(b)
# y.shape
# model = network.v5.sequence()
# y = model(b)
# y.shape
suggestion = network.v1.suggestion()
y = suggestion(batch)
y.shape
ll = y.split(1,1)
len(ll)

[i.squeeze(1).argmax(1) for i in y.split(1,1)]

# machine = network.v5.machine(model=model, device='cuda')
# machine.prepare()
# machine.learn(train=loader.train)

# b.keys()

# model = dict()


# import torch
# torch.tensor([1,2,3]).unsqueeze(0).split(1,1)[0]

