
from sklearn import preprocessing

class category:

    def encode(x='pandas column series'):

        engine = preprocessing.LabelEncoder()
        engine.fit(x)
        y = engine.transform(x)
        return(y)

    pass

