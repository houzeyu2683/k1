
from sklearn.model_selection import train_test_split, KFold

class split:

    def __init__(self, table=None, method='fold', size=10):

        self.table  = table
        self.method = method
        self.size   = size
        if(self.method=='fold'):

            knife = KFold(n_splits=self.size, random_state=0, shuffle=True)
            block = []
            for train, validation in knife.split(self.table): block += [{'train':train, 'validation': validation}]
            pass
        
        self.block  = block
        return

    def iterate(self):

        loop = range(self.size)
        return(loop)

    def get(self, fold=0):

        index = dict()
        index['train'], index['validation'] = self.block[fold]['train'], self.block[fold]['validation']
        self.train = self.table.iloc[index['train'],:].copy()
        self.validation = self.table.iloc[index['validation'],:].copy()
        return

    def clear(self):

        self.train = None
        self.validation = None
        return
    
    pass

# class tabulation:

#     def __init__(self, path):

#         self.path = path
#         return

#     def read(self):

#         self.data  = pandas.read_csv(self.path)
#         self.index = self.data['image'].unique().tolist()
#         self.size  = len(self.data)
#         return

#     def split(self, rate="(train, validation, test)"):

#         train, validation, test = slice(index=self.index, train=rate[0], validation=rate[1], test=rate[2])
#         self.train = self.data.loc[[i in train for i in self.data['image']]].copy().reset_index(drop=True)
#         self.validation = self.data.loc[[i in validation for i in self.data['image']]].copy().reset_index(drop=True)
#         self.test = self.data.loc[[i in test for i in self.data['image']]].copy().reset_index(drop=True)
#         return
    
#     def convert(self, what='train', to='dataset'):

#         if(to=='dataset'):

#             if(what=='data'): self.data = dataset(self.data)
#             if(what=='train'): self.train = dataset(self.train)
#             if(what=='validation'): self.validation = dataset(self.validation)
#             if(what=='test'): self.test = dataset(self.test)
#             pass

#         return
    
#     pass

