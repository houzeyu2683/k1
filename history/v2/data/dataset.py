
import torch
import pandas

class dataset:

    def __init__(self, data=None, train=None, validation=None, test=None):

        self.data       = data
        self.train      = train
        self.validation = validation
        self.test       = test
        if(not pandas.DataFrame(self.data).empty): self.data = generation(self.data)
        if(not pandas.DataFrame(self.train).empty): self.train = generation(self.train)
        if(not pandas.DataFrame(self.validation).empty): self.validation = generation(self.validation)
        if(not pandas.DataFrame(self.test).empty): self.test = generation(self.test)
        return
    
    pass

class generation(torch.utils.data.Dataset):

    def __init__(self, table):
        
        self.table = table.reset_index(drop=True)
        return

    def __getitem__(self, index):

        item = self.table.loc[index]
        return(item)
    
    def __len__(self):

        length = len(self.table)
        return(length)

    pass