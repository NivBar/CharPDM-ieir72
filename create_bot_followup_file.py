import pandas as pd
# group,query_id,username,text
from itertools import product
from config import current_prompt as cp

greg_data = pd.read_csv("greg_data.csv")
bots = ["MABOT", "MTBOT", "NMABOT", "NMTBOT"]
queries = greg_data["query_id"].unique()
rounds = list(greg_data["round_no"].unique())
rounds.remove(1)
gb_df = greg_data.groupby("query_id")

rows = []
for q_id, df_group in gb_df:
    users = df_group["username"].unique()
    for bot, creator in list(product(bots, users)):
        for r in rounds:
            if r == 2 and bot in ["NMABOT", "NMTBOT"]:
                continue
            rows.append({"round_no": r, "query_id": q_id, "creator": creator,"username": bot, "text": ""})

df = pd.DataFrame(rows).sort_values(["round_no","query_id","username"], ascending=[False,True,True])
try:
    text_df = pd.read_csv(f"bot_followup_{cp}.csv")
    df = pd.merge(df, text_df, how='inner', on=['round_no','query_id','creator','username']).drop('text_x', axis=1).rename(columns={'text_y':'text'})
except:
    pass
df.to_csv(f"bot_followup_{cp}.csv", index=False)
print(df)
