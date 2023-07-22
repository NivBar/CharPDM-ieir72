import pandas as pd

cols = ['username', 'query_id', 'text']
bot_data = pd.read_csv("bot_followup.csv")[cols]
greg_data = pd.read_csv("greg_data.csv").rename({"TEXT": "text"}, axis=1)[cols]
query_df = pd.read_csv("greg_data.csv")[['query_id','query_str']]
concat_df = pd.concat([bot_data,greg_data])
concat_df = concat_df.merge(query_df, how='left', on='query_id')
concat_df.to_csv("rank_data.csv", index=False)
x=1