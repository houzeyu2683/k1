
import os
import json
import pickle

class cache:

    def __init__(self, version='0.0.0'):
        
        self.version = version
        return

    def save(self, name, file=''):

        folder = os.path.dirname(file)
        os.makedirs(folder, exist_ok=True)
        with open(file, 'wb') as paper: pickle.dump(name, paper)
        return

    def load(self, file=''):

        with open(file, 'rb') as paper: variable = pickle.load(paper)
        return(variable)

    pass
