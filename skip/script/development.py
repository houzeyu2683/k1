
import library
import data

##  Load all data table.
table = data.table(source='preprocess')
split = data.split(table=table.f1, method='fold', size=10)

k = 1
split.get(fold=k)
dataset = data.dataset(train=split.train, validation=split.validation)



split.clear()




[i for i in split.train['truth'].head()]

generation(split.train).__getitem__(1)

# generation.train.shape
# generation.train.head(3)
# generation.validation.shape
# generation.validation.head(3)
# a = set(generation.validation['customer_id'])
# b = set(generation.validation['customer_id'])
# set.intersection(a,b)
# train, test = generation.block[0]
# table.user.loc[generation.block[0]['train']]
# table.user.tail()

# target = library.pandas.read_csv("resource/preprocess/csv/target.csv")
# user = library.pandas.read_csv("resource/preprocess/csv/user.csv")

# user['postal_code'].nunique()
# user['postal_code'].min()

# user['club_member_status'] = user['club_member_status'] / 4
# user['fashion_news_frequency'] = user['fashion_news_frequency'] / 5
# user['age'] = user['age'] / 100

# user.to_csv('user.csv', index=False)

# merge = library.pandas.concat([user, target], axis=1)

# merge = merge.loc[:,~merge.columns.duplicated()]
# merge.columns

# merge.to_csv("f1.csv", index=False)




