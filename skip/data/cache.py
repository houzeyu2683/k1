
import os

class cache:

    def __init__(self, storage='./'):
        
        self.storage = storage
        return
    
    def save(self, what=None, file=None, format='csv'):

        os.makedirs(self.storage, exist_ok=True)
        if(format=='csv'): what.to_csv(os.path.join(self.storage, file), index=False)
        return

    pass

