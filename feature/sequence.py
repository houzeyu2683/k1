

class sequence:

    def flatten(table="transaction", key='customer_id', variable='article_code', group='[customer_id, t_dat]'):

        y = table[group+[variable]].groupby(group)[variable].apply(" ".join).reset_index()
        y = y[group+[variable]].groupby(key)[variable].apply(" ".join).reset_index()
        return(y)

    pass
