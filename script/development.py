
##  Packages.
import v1

##
table = v1.data.table(source='resource/preprocess/csv')
table.load(file='group(train).csv')

##
split = v1.data.split(table=table.file, method='fold', size=20)
split.get(1)

##
dataset = v1.data.dataset(train=split.train, validation=split.validation, test=None)

##
item = dataset.train.__getitem__(0)
mode = 'train'
engine = v1.data.process(item, mode)
engine.prepare()
vector = engine.handle(step='vector')
sequence = engine.handle(step='sequence')

##
loader = v1.data.loader(batch=1)
loader.define(train=dataset.train)
batch = next(iter(loader.train))

##
vector = v1.network.vector()
vector(batch)

##
sequence = v1.network.sequence()
y = sequence(batch)

fusion = v1.network.fusion()
y = fusion(batch)

suggestion = v1.network.suggestion()
likelihood, prediction = suggestion(batch)

model = v1.network.model()
likelihood, prediction = model(batch)

##
import tqdm
import torch
import numpy
history = v1.network.history()
device = 'cuda'
cost = torch.nn.CrossEntropyLoss(ignore_index=0)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
metric = v1.network.metric(limit=12)
model.train()
iteration  = {
    'total loss':[],
    'map@12 score':[]
}
progress = tqdm.tqdm(loader.train, leave=False)
for batch in progress:
    
    model.zero_grad()
    target = 'article_id_code'
    vector = ['FN', 'Active', 'age', 'club_member_status', 'fashion_news_frequency', 'postal_code']
    sequence = [
        'price', 'sales_channel_id', 'product_code', 'prod_name', 'product_type_no', 
        'product_type_name', 'product_group_name', 'graphical_appearance_no', 
        'graphical_appearance_name', 'colour_group_code', 'colour_group_name', 
        'perceived_colour_value_id', 'perceived_colour_value_name', 'perceived_colour_master_id', 
        'perceived_colour_master_name', 'department_no', 'department_name', 'index_code', 
        'index_name', 'index_group_no', 'index_group_name', 'section_no', 'section_name', 
        'garment_group_no', 'garment_group_name', 'detail_desc', 'article_id_code'
    ]
    pass
    
    ##  Vector.
    for k in vector: 
        
        batch[k] = batch[k].to(device) 
        pass

    ##  Sequence.
    for k in sequence: 
        
        batch[k]['history'] = batch[k]['history'].to(device) 
        batch[k]['future'] = batch[k]['future'].to(device) 
        pass

    likelihood, prediction = model(batch)
    pass

    loss = 0.0
    loss += cost(likelihood.flatten(0,1), batch[target]['future'][1:,].flatten(0,1))
    loss.backward()
    optimizer.step()
    pass

    ##  Metric.
    score = 0.0
    truth = [i.split() for i in batch['item']['article_id_code']]
    score += metric.compute(prediction, truth)
    pass

    iteration['total loss'] += [round(loss.item(), 3)]
    iteration['map@12 score'] += [round(score, 3)]
    pass

    value = (
        iteration['total loss'][-1],
        iteration['map@12 score'][-1]
    )
    message = "[train] total loss : {} | map@12 score : {}".format(*value)
    progress.set_description(message)
    pass
    
history.loss['train'] += [round(numpy.array(iteration['total loss']).mean(), 3)]
history.metric['train'] += [round(numpy.array(iteration['map@12 score']).mean(), 3)]
pass

