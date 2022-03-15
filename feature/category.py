
from sklearn import preprocessing

class category:

    def encode(x='pandas column series', start=0):

        engine = preprocessing.LabelEncoder()
        engine.fit(x)
        y = engine.transform(x) + start
        return(y)

    pass

