
import data

table = data.table(source="resource/preprocess/sample/")
table.load(file="csv/feature(sample).csv")
table.file.head(4)

split = data.split(table=table.file, group=None, method='fold', size=4)
split.get(fold=0)
print(split.train.shape)
print(split.train.head())
print(split.validation.shape)
print(split.validation.head())

dataset = data.dataset(train=split.train, validation=split.validation)

loader = data.loader(batch=4, device='cuda')
loader.define(train=dataset.train, validation=dataset.validation)
batch = next(iter(loader.train))
# batch['r2'][0].shape
# batch['r2'][1].shape

# batch = next(iter(loader.validation))
# batch.keys()

# batch['future'][3]