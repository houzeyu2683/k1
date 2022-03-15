
import pandas
import functools
import numpy

class cross:

    def statistic(table=None, key='id', category=None, numeric=None):

        group = list()
        selection = table[[key, category, numeric]]
        pass

        method = 'sum'
        item = selection.groupby([key, category])[numeric].sum().round(2).reset_index()
        item = item.pivot_table(values=numeric, index=key, columns=category, aggfunc='first').reset_index()
        item = item.fillna(0)
        item.columns = [key] + ["{}_{}_{}".format(category, i, method) for i in item.columns[1:]]
        item.iloc[:,1:] = item.iloc[:,1:].apply(lambda x: x/x.max(), axis=0)
        group = group + [item]
        pass

        method = 'mean'
        item = selection.groupby([key, category])[numeric].mean().round(2).reset_index()
        item = item.pivot_table(values=numeric, index=key, columns=category, aggfunc='first').reset_index()
        item = item.fillna(0)
        item.columns = [key] + ["{}_{}_{}".format(category, i, method) for i in item.columns[1:]]
        item.iloc[:,1:] = item.iloc[:,1:].apply(lambda x: x/x.max(), axis=0)
        group = group + [item]
        pass

        method = 'max'
        item = selection.groupby([key, category])[numeric].max().round(2).reset_index()
        item = item.pivot_table(values=numeric, index=key, columns=category, aggfunc='first').reset_index()
        item = item.fillna(0)
        item.columns = [key] + ["{}_{}_{}".format(category, i, method) for i in item.columns[1:]]
        item.iloc[:,1:] = item.iloc[:,1:].apply(lambda x: x/x.max(), axis=0)
        group = group + [item]
        pass

        method = 'min'
        item = selection.groupby([key, category])[numeric].min().round(2).reset_index()
        item = item.pivot_table(values=numeric, index=key, columns=category, aggfunc='first').reset_index()
        item = item.fillna(0)
        item.columns = [key] + ["{}_{}_{}".format(category, i, method) for i in item.columns[1:]]
        group = group + [item]
        pass

        method = 'var'
        item = selection.groupby([key, category])[numeric].var().round(2).reset_index()
        item = item.pivot_table(values=numeric, index=key, columns=category, aggfunc='first').reset_index()
        item = item.fillna(0)
        item.columns = [key] + ["{}_{}_{}".format(category, i, method) for i in item.columns[1:]]
        item.iloc[:,1:] = item.iloc[:,1:].apply(lambda x: x/x.max(), axis=0)
        group = group + [item]
        pass

        group = functools.reduce(lambda x, y: pandas.merge(x, y, on=key, how='inner'), group)
        return(group)

    pass


