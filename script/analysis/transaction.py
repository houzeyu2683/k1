
from library import *

root = './resource/csv/'
start = time.time()
transaction = pandas.read_csv(os.path.join(root, "transactions_train.csv"), dtype={'article_id':str})
end = time.time()
transaction

'''
>>> transaction
               t_dat                                        customer_id  article_id     price  sales_channel_id
0         2018-09-20  000058a12d5b43e67d225668fa1f8d618c13dc232df0ca...  0663713001  0.050831                 2
1         2018-09-20  000058a12d5b43e67d225668fa1f8d618c13dc232df0ca...  0541518023  0.030492                 2
2         2018-09-20  00007d2de826758b65a93dd24ce629ed66842531df6699...  0505221004  0.015237                 2
3         2018-09-20  00007d2de826758b65a93dd24ce629ed66842531df6699...  0685687003  0.016932                 2
4         2018-09-20  00007d2de826758b65a93dd24ce629ed66842531df6699...  0685687004  0.016932                 2
...              ...                                                ...         ...       ...               ...
31788319  2020-09-22  fff2282977442e327b45d8c89afde25617d00124d0f999...  0929511001  0.059305                 2
31788320  2020-09-22  fff2282977442e327b45d8c89afde25617d00124d0f999...  0891322004  0.042356                 2
31788321  2020-09-22  fff380805474b287b05cb2a7507b9a013482f7dd0bce0e...  0918325001  0.043203                 1
31788322  2020-09-22  fff4d3a8b1f3b60af93e78c30a7cb4cf75edaf2590d3e5...  0833459002  0.006763                 1
31788323  2020-09-22  fffef3b6b73545df065b521e19f64bf6fe93bfd450ab20...  0898573003  0.033881                 2

[31788324 rows x 5 columns]
'''



