
import pandas, numpy
from sklearn.model_selection import train_test_split

class table:

    def __init__(self, path=None, target=None, split=0.1, seed=0):

        self.path = path
        self.target = target
        self.split = split
        self.seed = seed
        return
    
    def read(self):

        form = pandas.read_csv(self.path)
        data = form.loc[form['mode']!='test'].reset_index(drop=True)
        test = form.loc[form['mode']=='test'].reset_index(drop=True)
        pass
        
        numpy.random.seed(self.seed)
        if(self.target):

            train, exam = train_test_split(data, stratify=data[self.target], test_size=self.split)
            pass

        else:

            train, exam = train_test_split(data, test_size=self.split)
            pass

        train = train.reset_index(drop=True)
        exam  = exam.reset_index(drop=True)
        return(train, exam, test)

    # def split(dataframe, column, value):

    #     output = dataframe.loc[table[column]==value].copy()
    #     output = output.reset_index(drop=True)
    #     return(output)

    # pass








    # ##  Balance the data of table with target.
    # def balance(table, target, size):

    #     output = []
    #     for i in set(table[target]):

    #         selection = table[table[target]==i]
    #         pass
        
    #         if(len(selection)>size):

    #             selection = selection.sample(size)
    #             pass

    #         else:

    #             selection = selection.sample(size, replace=True)
    #             pass

    #         output = output + [selection]
    #         pass

    #     output = pandas.concat(output, axis=0)
    #     return(output)

    # ##
    # def unbalance(table, target, size):

    #     group = []
    #     for key, value in size.items():

    #         selection = table.loc[table[target]==key]
    #         pass

    #         if(len(selection)>value):
        
    #             group += [selection.sample(value)]
    #             pass
        
    #         else:
        
    #             group += [selection.sample(value, replace=True)]
    #             pass
        
    #     output = pandas.concat(group, axis=0)
    #     return(output)
