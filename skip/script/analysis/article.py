
from library import *

root = './resource/csv/'
start = time.time()
article = pandas.read_csv(os.path.join(root, "articles.csv"))
end = time.time()
article

'''
>>> article
        article_id  product_code               prod_name  product_type_no  ...            section_name garment_group_no  garment_group_name                                        detail_desc
0        108775015        108775               Strap top              253  ...  Womens Everyday Basics             1002        Jersey Basic            Jersey top with narrow shoulder straps.
1        108775044        108775               Strap top              253  ...  Womens Everyday Basics             1002        Jersey Basic            Jersey top with narrow shoulder straps.
2        108775051        108775           Strap top (1)              253  ...  Womens Everyday Basics             1002        Jersey Basic            Jersey top with narrow shoulder straps.
3        110065001        110065       OP T-shirt (Idro)              306  ...         Womens Lingerie             1017   Under-, Nightwear  Microfibre T-shirt bra with underwired, moulde...
4        110065002        110065       OP T-shirt (Idro)              306  ...         Womens Lingerie             1017   Under-, Nightwear  Microfibre T-shirt bra with underwired, moulde...
...            ...           ...                     ...              ...  ...                     ...              ...                 ...                                                ...
105537   953450001        953450  5pk regular Placement1              302  ...           Men Underwear             1021    Socks and Tights  Socks in a fine-knit cotton blend with a small...
105538   953763001        953763       SPORT Malaga tank              253  ...                    H&M+             1005        Jersey Fancy  Loose-fitting sports vest top in ribbed fast-d...
105539   956217002        956217         Cartwheel dress              265  ...            Womens Trend             1005        Jersey Fancy  Short, A-line dress in jersey with a round nec...
105540   957375001        957375        CLAIRE HAIR CLAW               72  ...     Divided Accessories             1019         Accessories                           Large plastic hair claw.
105541   959461001        959461            Lounge dress              265  ...            Womens Trend             1005        Jersey Fancy  Calf-length dress in ribbed jersey made from a...

[105542 rows x 25 columns]
'''
