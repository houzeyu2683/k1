
from sklearn import preprocessing

class label:

    def encode(x='[a, c, c, b]'):

        engine = preprocessing.LabelEncoder()
        engine.fit(x)
        y = engine.transform(x)
        return(y)

    pass
