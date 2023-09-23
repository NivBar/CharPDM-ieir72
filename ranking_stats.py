import os
from config import current_prompt as cp
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

file_path = f'/lv_local/home/niv.b/content_modification_code-master/Results/RankedLists/LambdaMART{cp}'

columns = ['query_id', 'Q0', 'docno', 'rank', 'score', 'method']

with open(file_path, 'r') as file:
 lines = file.readlines()

data = []

for line in lines:
    values = line.strip().split()
    query_id, _, document_id, rank, score, method = values
    query_id_padded = "{:03}".format(int(query_id))
    data.append([query_id_padded, 'Q0', document_id, int(rank), float(score), method])

df = pd.DataFrame(data, columns=columns).drop(["Q0", "method"], axis=1)

#### Features preprocess ####
folder_path = f'/lv_local/home/niv.b/content_modification_code-master/Results/Features/{cp}'

# Iterate over files in the folder
for filename in tqdm(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)
    feat, qid = file_path.split('/')[-1].split("_")
    if feat not in df.columns:
        df[feat] = np.nan
    with open(file_path, 'r') as file:
        lines = file.readlines()

 # Process each line and extract the values for each column
for line in lines:
    values = line.strip().split()
    try:
        docno, score = values
        df.loc[df['docno'] == docno, feat] = score
    except:
        docno, sum_, min_, max_, mean_, var_ = values
        df.loc[df['docno'] == docno, feat] = mean_

df.to_csv(f"feature_data_{cp}.csv", index=False)

#### new features ####
df = pd.read_csv(f"feature_data_{cp}.csv").drop("rank", axis=1)
df[["round_no", "query_id", "username", "creator"]] = df["docno"].apply(lambda x: pd.Series(x.split("-")))
maximal_epoch, minimal_epoch = df.round_no.max(), df.round_no.min()
df.round_no, df.query_id = df.round_no.astype(int), df.query_id.astype(int)

greg_data = pd.read_csv("greg_data.csv")[['round_no', 'query_id', 'username', 'position']].rename(
    {"position": "original_position"}, axis=1)
greg_data.round_no, greg_data.query_id, greg_data.username = greg_data.round_no.astype(int), greg_data.query_id.astype(
    int), greg_data.username.astype(str)
df = df.merge(greg_data, on=['round_no', 'query_id', 'username'], how='left')

prev_round = []
for _, row in greg_data[greg_data.round_no == int(minimal_epoch) - 1].iterrows():
    query_str = '0' + str(row.query_id)
    docno = f"0{int(minimal_epoch) - 1}-{query_str[-3:]}-{row.username}-creator"
    prev_round.append(
        {"round_no": row.round_no, "query_id": row.query_id, "creator": "creator", "username": row.username,
         "docno": docno, "original_position": row.original_position})
prev_df = pd.DataFrame(prev_round)
df = pd.concat([prev_df, df]).reset_index(drop=True)
df.round_no, df.query_id = df.round_no.astype(int), df.query_id.astype(int)
df.set_index("docno", inplace=True)

# first_round = []
# for _, row in greg_data[greg_data.round_no == 1].iterrows():
#     query_str = '00' + str(row.query_id)
#     docno = f"01-{query_str[-3:]}-{row.username}-creator"
#     first_round.append(
#         {"round_no": row.round_no, "query_id": row.query_id, "creator": "creator", "username": row.username,
#          "docno": docno, "original_position": row.original_position})
# first_df = pd.DataFrame(first_round)
# df = pd.concat([first_df, df]).reset_index(drop=True)
# df.round_no, df.query_id = df.round_no.astype(int), df.query_id.astype(int)
# df.set_index("docno", inplace=True)

# Looping through each row in the DataFrame
for idx, row in tqdm(df.iterrows()):

    # If it's the first round, set 'previous_docno' to NaN and move to the next row
    # if row.round_no == 1:
    #     df.loc[idx, "previous_docno"] = np.nan
    #     continue

    if not int(minimal_epoch) <= row.round_no <= int(maximal_epoch):
        df.loc[idx, "previous_docno"] = np.nan
        continue

    if row.creator != "creator":  # bot docs
        prev_docno = df.index[
            (df.round_no == row.round_no - 1) & (df.query_id == row.query_id) & (df.creator == "creator") & (
                    df.username == row.creator)].tolist()[0]
        orig_docno = df.index[
            (df.round_no == row.round_no) & (df.query_id == row.query_id) & (df.creator == "creator") & (
                    df.username == row.creator)].tolist()[0]
        x = 1
        df.at[idx, "original_position"] = df[df.index == orig_docno].original_position.values[0]
    else:  # student docs
        # if row.round_no == int(minimal_epoch):
        #     df.loc[idx, "previous_docno"] = np.nan
        #     continue

        prev_docno = df.index[
            (df.round_no == row.round_no - 1) & (df.query_id == row.query_id) & (df.creator == "creator") & (
                    df.username == row.username)].tolist()[0]
        x = 1

    # Set the 'previous_docno' and 'previous_pos' for the current row based on the found 'previous_docno'
    df.loc[idx, "previous_docno"] = prev_docno
    df.loc[idx, "previous_pos"] = df[df.index == prev_docno].original_position.values[0]

# Grouping the DataFrame by 'round_no' and 'query_id' for further processing
df_gb = df.groupby(["round_no", "query_id"])

# Looping through each group
for group_name, df_group in tqdm(df_gb):
    # Splitting the group into 'non_bots' (where 'creator' is "creator") and 'bots' (where 'creator' is not "creator")
    if group_name[0] not in list(range(int(minimal_epoch), int(maximal_epoch) + 1)):
        continue
    non_bots = df_group[df_group.creator == "creator"]
    bots = df_group[df_group.creator != "creator"]

    # Looping through each row in 'bots' group to calculate 'current_pos' for each bot
    for idx, row in bots.iterrows():
        # Concatenating the 'non_bots' group and the current bot row to create a comparison DataFrame
        comp_df = pd.concat([non_bots, row.to_frame().T])

        # Removing the current bot from the comparison DataFrame
        comp_df = comp_df[(comp_df.username != row.creator) | (comp_df.creator == row.username)]

        # Calculating ranks based on the 'score' column in the comparison DataFrame
        comp_df['rank'] = comp_df['score'].rank(ascending=False, method='dense')

        # Setting the 'current_pos' for the current bot in the main DataFrame
        assert comp_df.loc[idx, "rank"] % 1 == 0
        df.loc[idx, "current_pos"] = comp_df.loc[idx, "rank"]

        # if row.creator == "BOT":
        #     df.loc[idx, "BOT_current_pos"] = df.loc[idx, "current_pos"]
        #     df.loc[idx, "win_over_BOT"] = np.nan
        #     continue
        # df.loc[idx, "BOT_current_pos"] = comp_df.loc[comp_df[comp_df.username == 'BOT'].index[0], "rank"] if not \
        #     comp_df[comp_df.username == 'BOT'].empty else np.nan
        # df.loc[idx, "win_over_BOT"] = True if df.loc[idx, "current_pos"] <= df.loc[idx, "BOT_current_pos"] else False

df.loc[df.creator == 'creator', 'current_pos'] = df.loc[df.creator == 'creator', "original_position"]

df['pos_diff'] = df.apply(
    lambda row: row['current_pos'] - row['previous_pos'] if pd.notna(row['current_pos']) and pd.notna(
        row['previous_pos']) else np.nan, axis=1)


df['scaled_pos_diff'] = np.where(pd.isna(df['pos_diff']), np.nan,
                                 np.where(df['pos_diff'] > 0, df['pos_diff'] / (5 - df['previous_pos']),
                                          df['pos_diff'] / (df['previous_pos'] - 1)))
df['scaled_pos_diff'] = np.where(pd.notna(df['scaled_pos_diff']), df['scaled_pos_diff'],
                                 np.where(df['pos_diff'] == 0, 0,
                                          np.nan))
# addition

# df['BOT_pos_diff'] = df.apply(
#     lambda row: row['BOT_current_pos'] - row['BOT_previous_pos'] if pd.notna(row['BOT_current_pos']) and pd.notna(
#         row['BOT_previous_pos']) else np.nan, axis=1)
#
# df['BOT_scaled_pos_diff'] = np.where(pd.isna(df['BOT_pos_diff']), np.nan,
#                                  np.where(df['BOT_pos_diff'] > 0, df['BOT_pos_diff'] / (5 - df['BOT_previous_pos']),
#                                           df['BOT_pos_diff'] / (df['BOT_previous_pos'] - 1)))
# df['BOT_scaled_pos_diff'] = np.where(pd.notna(df['BOT_scaled_pos_diff']), df['BOT_scaled_pos_diff'],
#                                  np.where(df['BOT_pos_diff'] == 0, 0,
#                                           np.nan))
# addition


# Resetting the index of the DataFrame
df.reset_index(inplace=True)

# Updating the 'original_position' column based on the 'previous_docno'
# df["original_position"] = df.apply(
#     lambda row: df[df.docno == row.docno.replace(row.creator, 'creator').replace(row.username,
#                                                                                  row.creator)].original_position.values[
#         0] if pd.isna(row.original_position) and not
#     df[df.docno == row.docno.replace(row.creator, 'creator').replace(row.username,
#                                                                      row.creator)].empty else row.original_position,
#     axis=1)


# TODO: pay attention to this line
# df.loc[df.original_position.isna(), 'original_position'] = df[df.original_position.isna()].apply(
#     lambda row: df.loc[df.docno == row.docno.replace('-BOT', '-creator').replace(row.username,
#                                                                                  row.creator), 'original_position'].values[
#         0], axis=1)

# def fill_original_position(row):
#     target_docno = row.docno.replace(row.username, row.creator)
#     target_values = df.loc[df.docno == target_docno, 'original_position'].values

    # if len(target_values) == 0:
    #     print(f"No match found for docno: {target_docno}")
    #     return np.nan  # or some other value that makes sense in your context
    #
    # return target_values[0]


# df.loc[df.original_position.isna(), 'original_position'] = df[df.original_position.isna()].apply(fill_original_position,
#                                                                                                  axis=1)

# df.loc[df.creator == 'BOT', 'win_over_BOT'] = np.where(df[df.creator == 'BOT']['current_pos'] - df[df.creator == 'BOT']['original_position'] <= 0, True, False)

# Calculating 'scaled_orig_pos_diff' based on 'orig_pos_diff', considering some conditions
df['orig_pos_diff'] = df.apply(
    lambda row: row['current_pos'] - row['original_position'] if pd.notna(row['current_pos']) and pd.notna(
        row['original_position']) else np.nan, axis=1)

df['scaled_orig_pos_diff'] = np.where(pd.isna(df['orig_pos_diff']), np.nan,
                                      np.where(df['orig_pos_diff'] > 0,
                                               df['orig_pos_diff'] / (5 - df['original_position']),
                                               df['orig_pos_diff'] / (df['original_position'] - 1)))

df['scaled_orig_pos_diff'] = np.where(pd.isna(df['scaled_orig_pos_diff']),
                                      np.where(df['orig_pos_diff'] == 0, 0, np.nan),
                                      df['scaled_orig_pos_diff'])

# df['scaled_orig_pos_diff'] = np.where(pd.notna(df['scaled_orig_pos_diff']), df['scaled_pos_diff'],
#                                       np.where(df['orig_pos_diff'] == 0, 0,
#                                                np.nan))

# Save the final DataFrame to a new CSV file "feature_data_4_new.csv"
df = df[df.round_no != 1]
df = df[df.score.notna()]

#TODO: remove this line if condition change - removing docs ranked first
df = df[df.previous_pos != 1]

df.to_csv(f"feature_data_{cp}_new.csv", index=False)