
import torch
x = torch.nn.functional.one_hot(torch.tensor([0,1,2,1]), 400000)


# import library
# import data
# import network

# ##  Load all data table.
# table = data.table(source='preprocess')
# fold  = 20
# split = data.split(table=table.f1, method='fold', size=fold)
# split.get(fold=1)
# dataset = data.dataset(train=split.train, validation=split.validation)
# loader = data.loader(batch=7)
# loader.define(train=dataset.train, validation=dataset.validation)

# batch = next(iter(loader.train))



# table.f1



# batch['sequence(t_number)']['history'][:,:,0]
# batch.keys()
# batch['vector(numeric)'].shape
# batch['vector(club_member_status)'].shape
# batch['vector(fashion_news_frequency)'].shape
# batch['vector(postal_code)'].shape
# batch['vector(club_member_status)'].shape
# batch['vector(fashion_news_frequency)'].shape
# batch['vector(postal_code)'].shape
# batch['sequence(price)']['history'].shape
# batch['sequence(article_code)']['history'].shape
# batch['sequence(article_code_delta)']['history'].shape


# batch['sequence(t_number)']['history'].shape





# import torch
# from torch.nn.utils import rnn
# torch.cat(batch['vector(numeric)'], 0).shape
# batch['vector(category)'] = torch.cat(batch['vector(category)'], 1)
# batch['vector(category)'].split(1)[0].shape
# batch['sequence(price)']['history']
# rnn.pad_sequence(batch['sequence(price)']['history'], batch_first=False, padding_value=0).shape
# rnn.pad_sequence(batch['sequence(article_code)']['history'], batch_first=False, padding_value=0).shape
# batch['sequence(article_code)']

# ##  Each fold.
# score = {'train':[], "validation":[]}
# # k = 0
# for k in split.iterate():

#     print("start fold {}".format(k))
#     split.get(fold=k)
#     dataset = data.dataset(train=split.train, validation=split.validation)
#     pass

#     loader = data.loader(batch=36)
#     loader.define(train=dataset.train, validation=dataset.validation)
#     pass

#     model = network.v5.model()
#     machine = network.machine(model=model, device='cuda', folder='log/fold({})'.format(k))
#     machine.prepare()
#     pass

#     epoch = 10
#     # e = 0
#     # batch = next(iter(loader.train))
#     # o = model(batch)
#     # o['embedding(article_code)'].shape
#     # batch['sequence(article_code)']['future'].shape
#     for e in range(epoch):

#         machine.learn(train=loader.train,validation=loader.validation)
#         machine.save(what='history')
#         machine.save(what='checkpoint', mode='better')
#         machine.update(what='checkpoint')
#         continue
    
#     score['train'] += [max(machine.history.metric['train'])]
#     score['validation'] += [max(machine.history.metric['validation'])]
#     pass

# value = library.numpy.mean(score['train']), library.numpy.mean(score['validation'])
# print("[train] map@12 : {} | [validation] map@12 : {}".format(*value))
# pass



# # import library
# # import data
# # import network

# # ##  Load all data table.
# # table = data.table(source='preprocess')
# # table.f1 = table.f1.loc[table.f1['seq_len']<10]

# # fold = 20
# # split = data.split(table=table.f1, method='fold', size=fold)

# # ##  Each fold.
# # score = {'train':[], "validation":[]}
# # # k = 0
# # for k in split.iterate():

# #     print("start fold {}".format(k))
# #     split.get(fold=k)
# #     dataset = data.dataset(train=split.train, validation=split.validation)
# #     pass

# #     loader = data.loader(batch=36)
# #     loader.define(train=dataset.train, validation=dataset.validation)
# #     pass

# #     model = network.v5.model()
# #     machine = network.machine(model=model, device='cuda', folder='log/fold({})'.format(k))
# #     machine.prepare()
# #     pass

# #     epoch = 10
# #     # e = 0
# #     # batch = next(iter(loader.train))
# #     # o = model(batch)
# #     # o['embedding(article_code)'].shape
# #     # batch['sequence(article_code)']['future'].shape
# #     for e in range(epoch):

# #         machine.learn(train=loader.train,validation=loader.validation)
# #         machine.save(what='history')
# #         machine.save(what='checkpoint', mode='better')
# #         machine.update(what='checkpoint')
# #         continue
    
# #     score['train'] += [max(machine.history.metric['train'])]
# #     score['validation'] += [max(machine.history.metric['validation'])]
# #     pass

# # value = library.numpy.mean(score['train']), library.numpy.mean(score['validation'])
# # print("[train] map@12 : {} | [validation] map@12 : {}".format(*value))
# # pass
