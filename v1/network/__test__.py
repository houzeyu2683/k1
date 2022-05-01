
import data

table = data.table(source="resource/preprocess/sample/")
table.load(file="csv/feature(sample).csv")
table.file.head(4)

split = data.split(table=table.file, group=None, method='fold', size=10)
split.get(fold=0)

dataset = data.dataset(train=split.train, validation=split.validation, test=None)

loader = data.loader(batch=2, device='cuda')
loader.define(train=dataset.train, validation=dataset.validation)

batch = next(iter(loader.train))

import v1

model = v1.network.model(device='cuda')
machine = v1.network.machine(model=model, device='cuda', folder='log')
machine.prepare()

for epoch in range(20):

    machine.learn(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    machine.save(what='checkpoint')
    machine.update(what='checkpoint')
    pass

# y = model.forward(batch)
# loss = model.cost(batch)

# # personality = network.personality(device='cuda')
# # personality(batch).shape
# # behavior = network.behavior(device='cuda')
# # behavior(batch).shape
# model = network.model(device='cuda')
# # loss = model.cost(batch)
# # loss.backward()

# import torch
# import tqdm
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# for batch in tqdm.tqdm(loader.train):

#     optimizer.zero_grad()
#     score, prediction, target = model.forward(batch, inference=False)
#     loss = model.cost(score, target)
#     print(loss)
#     loss.backward()
#     optimizer.step()
#     pass
