import pandas as pd
# group,query_id,username,text
from itertools import product
from config import current_prompt as cp

greg_data = pd.read_csv("greg_data.csv")
bots = ["MABOT", "MTBOT", "NMABOT", "NMTBOT"]
queries = greg_data["query_id"].unique()

rows = []
for bot, q_id in list(product(bots, queries)):
    rows.append({"group": "A", "query_id": q_id, "username": bot, "text": ""})

df = pd.DataFrame(rows).sort_values(["query_id","username"])
df.to_csv(f"bot_followup_{cp}.csv", index=False)
print(df)
