
import pandas
import time
import os
from sklearn import preprocessing

class table:
    
    def __init__(self, source="/resource/preprocess/csv/"):

        self.source = source
        return

    def load(self, file=None):

        self.file = pandas.read_csv(os.path.join(self.source, file), dtype=str)
        return

    pass

