
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

class split:

    def __init__(self, table=None, group=None, method='fold', size=10):

        self.table  = table
        self.method = method
        self.size   = size
        self.group  = group
        if(self.method=='fold'):

            if(group==None):

                knife = KFold(n_splits=self.size, random_state=0, shuffle=True)
                block = []
                for train, validation in knife.split(self.table): block += [{'train':train, 'validation': validation}]
                pass
            
            else:

                knife = StratifiedKFold(n_splits=self.size, random_state=0, shuffle=True)
                block = []
                for train, validation in knife.split(self.table, self.table[self.group]): 
                    
                    block += [{'train':train, 'validation': validation}]
                    pass
                
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
