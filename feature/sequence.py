

class sequence:

    def flatten(table="transaction", key='customer_id', group='[customer_id, t_dat]', variable='article_code'):

        y = table[group+[variable]].groupby(group)[variable].apply(" ".join).reset_index()
        y = y[group+[variable]].groupby(key)[variable].apply(" ".join).reset_index()
        return(y)

    pass

# c = library.pandas.merge(table.transaction, cache.article[["article_id", l]], on="article_id", how='inner').copy()
# c[l] = c[l].astype(str)
# row[l] = c[['customer_id', 't_dat', l]].groupby(['customer_id', 't_dat'])[l].apply(" ".join).reset_index()
# row[l] = row[l][['customer_id', 't_dat', l]].groupby(['customer_id'])[l].apply(" ".join).reset_index()
