
import os
import json

class cache:

    def __init__(self, version=None):
        
        self.version = version
        return
    
    # def save(self, what=None, file=None, format='csv'):

    #     os.makedirs(os.path.join(self.storage, format), exist_ok=True)
    #     if(format=='csv'): what.to_csv(os.path.join(self.storage, format, file), index=False)
    #     if(format=='json'): 

    #         with open(os.path.join(self.storage, format, file), 'w') as paper: json.dump(what, paper)
    #         pass

    #     return

    pass

