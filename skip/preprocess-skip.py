
##  ====================================================================================================
##  The packages.
import feature
import library

##  The root of project.
##  Initialize the cache object for save the miscellany.
root = library.os.getcwd()
table = library.cache()
table.folder = library.os.path.join(library.os.getcwd(), 'resource/kaggle(sample)/csv')

##  ====================================================================================================
##  Load <article> data.
table.article = library.pandas.read_csv(library.os.path.join(table.folder, "articles.csv"), dtype=str)

##  Handle missing value.
table.article["detail_desc"] = table.article["detail_desc"].fillna("<NA>")

##  Label encoding for category variables.
key  = 'article_id_code'
head = 10
table.article[key] = feature.label.encode(table.article['article_id']) + head
loop = [
    'product_code', 'prod_name', "product_type_no", 'product_type_name',
    "product_group_name", "graphical_appearance_no", "graphical_appearance_name",
    "colour_group_code", 'colour_group_name', 'perceived_colour_value_id', 
    'perceived_colour_value_name', 'perceived_colour_master_id', 
    'perceived_colour_master_name', 'department_no', 'department_name', 
    'index_code', 'index_name', 'index_group_no', 'index_group_name', 
    'section_no', 'section_name', 'garment_group_no', 
    'garment_group_name', 'detail_desc'
]
for key in loop:

    table.article[key] = feature.label.encode(table.article[key]) + head
    value = table.article[key].nunique()
    print("{}:{}".format(key, value))
    pass

##  ====================================================================================================
##  Load <customer> table.
table.customer = library.pandas.read_csv(library.os.path.join(table.folder, "customers.csv"), dtype=str)

##  Handle missing value.
table.customer['FN'] = table.customer['FN'].fillna(0.0)
table.customer['Active'] = table.customer['Active'].fillna(0.0)
table.customer['club_member_status'] = table.customer['club_member_status'].fillna("<NA>")
table.customer['fashion_news_frequency'] = table.customer['fashion_news_frequency'].fillna("<NA>")
table.customer['age'] = table.customer['age'].fillna(-1)

##  Label encoding for category variables.
loop = ['club_member_status', 'fashion_news_frequency', 'postal_code']
head = 10
for key in loop:

    table.customer[key] = feature.label.encode(table.customer[key]) + head
    value = table.customer[key].nunique()
    print("{}:{}".format(key, value))    
    pass

##  ====================================================================================================
##  Load <transaction> table.
table.transaction = library.pandas.read_csv(library.os.path.join(table.folder, "transactions_train.csv"), dtype=str, nrows=100000)
table.transaction['t_dat'] = library.pandas.to_datetime(table.transaction['t_dat'])

##  Label encoding for category variables.
##  Transform numeric variable.
head = 10
table.transaction['t_dat_code'] = feature.label.encode(table.transaction['t_dat']) + head
table.transaction['sales_channel_id'] = feature.label.encode(table.transaction['sales_channel_id']) + head
table.transaction['price'] = [head + float(i) for i in table.transaction['price']]

# ##  ====================================================================================================
# ##  Save the checkpoint.
# storage = library.os.path.join(root, 'resource/{}/csv/'.format("clean"))
# library.os.makedirs(storage, exist_ok=True)
# table.article.to_csv(library.os.path.join(storage, 'article.csv'), index=False)
# table.customer.to_csv(library.os.path.join(storage, 'customer.csv'), index=False)
# table.transaction.to_csv(library.os.path.join(storage, 'transaction.csv'), index=False)

##  ====================================================================================================
##  Preprocess the tables to sequence by user.
table.sequence = dict()
loop = ['price', 'sales_channel_id', "t_dat_code"]
for variable in loop:

    table.sequence[variable] = feature.sequence.flatten(table=table.transaction.astype(str), key='customer_id', variable=variable, group=['customer_id', 't_dat'])
    pass

loop = [
    'product_code', 'prod_name', "product_type_no", 'product_type_name',
    "product_group_name", "graphical_appearance_no", "graphical_appearance_name",
    "colour_group_code", 'colour_group_name', 'perceived_colour_value_id', 
    'perceived_colour_value_name', 'perceived_colour_master_id', 
    'perceived_colour_master_name', 'department_no', 'department_name', 
    'index_code', 'index_name', 'index_group_no', 'index_group_name', 
    'section_no', 'section_name', 'garment_group_no', 
    'garment_group_name', 'detail_desc', 'article_id_code'
]
for variable in library.tqdm.tqdm(loop, total=len(loop)):

    selection = table.transaction[['t_dat', "customer_id", "article_id"]].copy()
    selection = library.pandas.merge(selection, table.article[["article_id", variable]], on="article_id", how='inner')
    table.sequence[variable] = feature.sequence.flatten(table=selection.astype(str), key='customer_id', group=['customer_id', 't_dat'], variable=variable)
    pass

merge = lambda x,y: library.pandas.merge(left=x, right=y, on='customer_id', how='inner')
table.sequence = library.functools.reduce(merge, table.sequence.values())

##  ====================================================================================================
##  Mix together and save the checkpoint.
storage = library.os.path.join(root, 'resource/{}/csv/'.format("preprocess"))
library.os.makedirs(storage, exist_ok=True)
table.group = library.pandas.merge(left=table.customer, right=table.sequence, on='customer_id', how='outer')
table.group.dropna().to_csv(library.os.path.join(storage, "group(train).csv"), index=False)
table.group.fillna("").to_csv(library.os.path.join(storage, "group(all).csv"), index=False)

# ##  ====================================================================================================
# ##  Generatoe the <edge> table between [article_id_code] and [article_id_code].
# loop  = " ".join(table.group['article_id_code'].dropna()).split()
# edge = ["1-{}".format(i) for i in set(loop)]
# total = len(loop)-1
# for a, b in library.tqdm.tqdm(zip(loop[:-1], loop[1:]), total=total):

#     edge = edge + ['-'.join([a,b])]
#     pass

# edge = library.pandas.DataFrame({"pair": edge})
# edge = edge.drop_duplicates()
# head = 10
# edge['pair_code'] = feature.label.encode(edge['pair']) + head
# table.edge = edge

# ##  Save the checkpoint.
# storage = library.os.path.join(root, 'resource/{}/csv/'.format("preprocess"))
# library.os.makedirs(storage, exist_ok=True)
# table.edge.to_csv(library.os.path.join(storage, "edge.csv"), index=False)
# table.edge.nunique()

# ##  Update to <group> table.
# for _, item in table.group.dropna().iterrows():

#     line = item['article_id_code'].split()
#     if(len(line)>1):

#         track = []
#         for a, b in zip(line[:-1], line[1:]):

#             code = table.edge.loc[table.edge['pair']=="-".join([a,b])]['pair_code'].item()
#             track += [str(code)]
#             pass
        
#         track = " ".join(track)
#         pass
    
#     if(len(line)==1):

#         code = table.edge.loc[table.edge['pair']=='1-{}'.format(line[-1])]['pair_code'].item()
#         track = [str(code)]
#         track = " ".join(track)
#         pass

#     break


