

class sequence:

    def flatten(table="transaction", key='customer_id', group='[customer_id, t_dat]', variable='article_code'):

        y = table[group+[variable]].groupby(group)[variable].apply(" ".join).reset_index()
        y = y[group+[variable]].groupby(key)[variable].apply(" ".join).reset_index()
        return(y)

    # def delta(sequence='41155 47422 47422 47645 41364'):
    
    #     sequence = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in sequence.split()]
    #     head = sequence[:-1]
    #     tail = sequence[1:]
    #     record = []
    #     for h, t in zip(head, tail): record += [str((t - h).days)]
    #     record = ' '.join(record)
    #     return(record)

    pass

# c = library.pandas.merge(table.transaction, cache.article[["article_id", l]], on="article_id", how='inner').copy()
# c[l] = c[l].astype(str)
# row[l] = c[['customer_id', 't_dat', l]].groupby(['customer_id', 't_dat'])[l].apply(" ".join).reset_index()
# row[l] = row[l][['customer_id', 't_dat', l]].groupby(['customer_id'])[l].apply(" ".join).reset_index()
