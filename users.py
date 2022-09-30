import pandas as pd
from utils import *

analysis_unit = 'users'

csv_file = f'./data/xhs/{analysis_unit}_clean.csv'
print('opening...:',csv_file)
df = pd.read_csv(csv_file,
                 index_col='index',
                 parse_dates=['post_date','crawl_date'])

df = convert_key_tag_top_novelty(df)

for novel_col in ['tags']:
    for kol_level in range(5):
        for time_int in ['d']:
            df_ = df[df['kol_level']==kol_level]
            df_ = get_novelty(df_,novel_col,time_int)
            df_.to_csv(f'./results/{analysis_unit}_\
{kol_level}_{novel_col}_{time_int}.csv')