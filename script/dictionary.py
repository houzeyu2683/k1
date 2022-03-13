
import library

article = library.pandas.read_csv("resource/kaggle/csv/articles.csv")
article['detail_desc'] = article['detail_desc'].fillna("missing value")
article




article.isna().sum()
