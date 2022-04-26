
import data

table = data.table(source="resource/preprocess/sample/")
table.load(file="csv/feature(train).csv")
table.file.head(4)

split = data.split(table=table.file, group=None, method='fold', size=10)
split.get(fold=0)

dataset = data.dataset(train=split.train, validation=split.validation, test=None)

loader = data.loader(batch=4)
loader.define(train=dataset.train)

batch = next(iter(loader.train))
batch

key = 'article_label'
batch[key]['future'].shape





