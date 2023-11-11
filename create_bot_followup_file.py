import glob

import pandas as pd
# group,query_id,username,text
from itertools import product

from tqdm import tqdm

from config import current_prompt as cp, ACTIVE_BOTS

greg_data = pd.read_csv("greg_data.csv")
greg_data = greg_data[(greg_data.round_no == 6) & (greg_data.position.between(2,5))] #TODO: training setting from the article
bots = ACTIVE_BOTS # bots using prompt bank
queries = greg_data["query_id"].unique()

rounds = list(greg_data["round_no"].unique())
if 1 in rounds: rounds.remove(1)
gb_df = greg_data.groupby("query_id")

rows = []
for q_id, df_group in tqdm(gb_df):
    users = df_group["username"].unique()
    for bot, creator in list(product(bots, users)):
        for r in rounds:
            rel_users = df_group[df_group["round_no"] == r]["username"].unique()
            if creator not in rel_users:
                continue
            # if r == 2 and bot in ["NMABOT", "NMTBOT"]:
            #     continue
            rows.append({"round_no": r, "query_id": q_id, "creator": creator,"username": bot, "text": ""})

final_df = pd.DataFrame(rows).sort_values(["round_no","query_id"], ascending=[False,True]).drop_duplicates()
x = 1
# try:
#     text_df = pd.read_csv(f"bot_followup_{cp}.csv")
#     df = pd.merge(df, text_df, how='inner', on=['round_no','query_id','creator','username']).drop('text_x', axis=1).rename(columns={'text_y':'text'})
# except:
#     pass
# asrc_df = pd.concat([pd.read_csv(f) for f in glob.glob('/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/bot_followup_asrc*.csv')], ignore_index=True)
# asrc_df = asrc_df[asrc_df.round_no.isin([6,7])]

# merged_df = pd.merge(df, asrc_df, on=['round_no', 'query_id', 'creator'], how='outer', indicator=True)
# merged_df = merged_df[['round_no', 'query_id', 'creator','_merge']].drop_duplicates()[merged_df._merge != 'both']
# merged_df = pd.merge(merged_df, greg_data, left_on=['query_id', 'round_no', 'creator'], right_on=['query_id', 'round_no', 'username'], how='left').sort_values(["round_no","query_id"])
# final_df = pd.concat([df, asrc_df], ignore_index=True).sort_values(["query_id", "username"])
# final_df = pd.merge(final_df,merged_df.drop("_merge",axis=1), indicator=True, how='outer', on=["round_no","query_id","creator"]).query('_merge=="left_only"').drop('_merge', axis=1).rename({"username_x":"username"}, axis=1)
# final_df = final_df[["round_no","query_id","creator","username","text"]]

# # incorporate previous versions' texts
# keep_texts = False
# if keep_texts:
#     filtered_final_df = final_df[final_df.username != 'BOT']
#     filtered_csv_df = pd.read_csv(f"bot_followup_{cp - 1}.csv")[lambda x: x.username != 'BOT']
#     merged_df = pd.merge(filtered_final_df, filtered_csv_df, how='left',
#                          on=['round_no', 'query_id', 'creator', 'username']).drop('text_x', axis=1).rename(
#         columns={'text_y': 'text'})
#     final_df.loc[final_df.username != 'BOT', 'text'] = merged_df['text']

#TODO: testing, delete to use all queries
queries = queries[:3]
final_df = final_df[final_df.query_id.isin(queries)]

final_df.to_csv(f"bot_followup_{cp}.csv", index=False)
print(final_df)
