import os
from config import current_prompt as cp
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

if not os.path.exists(f"feature_data_{cp}.csv"):
    print("Part 1 started")
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

    ### Features preprocess ####
    folder_path = f'/lv_local/home/niv.b/content_modification_code-master/Results/Features/{cp}'

    # Iterate over files in the folder
    for filename in tqdm(os.listdir(folder_path), desc="Processing files", total=len(os.listdir(folder_path)),
                         miniters=100):
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
                    try:
                        docno, sum_, min_, max_, mean_, var_ = values
                        df.loc[df['docno'] == docno, feat] = mean_
                    except:
                        x = 1

    df.to_csv(f"feature_data_{cp}.csv", index=False)
    print("Part 1 ended")

#### new features ####
if not os.path.exists(f"feature_data_{cp}_new.csv"):
    print("Part 2 started")
    df = pd.read_csv(f"feature_data_{cp}.csv").rename({"query_id": "query_id_new"}, axis=1)
    df = df[[col for col in df.columns if not col.startswith("doc")] + ['docno']]
    df.query_id_new = df.query_id_new.astype(int)
    # df = pd.read_csv(f"feature_data_{cp}.csv").drop("rank", axis=1)

    # TODO: assuming we run asrcqrels only!
    if 'qrels' in cp:
        df['query_id_new'] = df['query_id_new'].astype(str)
        df[["round_no", "query_id", "username", "creator"]] = df["query_id_new"].apply(
            lambda x: pd.Series([x[0], x[1:4], x[4:6], x[6:]]))
        df = df[~df.docno.str.contains("creator")]


    else:
        df[["round_no", "query_id", "username", "creator"]] = df["docno"].apply(lambda x: pd.Series(x.split("-")))

    maximal_epoch, minimal_epoch = df.round_no.max(), df.round_no.min()
    df.round_no, df.query_id = df.round_no.astype(int), df.query_id.astype(int)

    greg_data = pd.read_csv("greg_data.csv")[['round_no', 'query_id', 'username', 'position']].rename(
        {"position": "original_position"}, axis=1)
    greg_data.round_no, greg_data.query_id, greg_data.username = greg_data.round_no.astype(
        int), greg_data.query_id.astype(
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
    # df.set_index("docno", inplace=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", miniters=100):
        try:
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
                df.at[idx, "original_position"] = df[df.index == orig_docno].original_position.values[0]
            else:  # student docs
                prev_docno = df.index[
                    (df.round_no == row.round_no - 1) & (df.query_id == row.query_id) & (df.creator == "creator") & (
                            df.username == row.username)].tolist()[0]

            # Set the 'previous_docno' and 'previous_pos' for the current row based on the found 'previous_docno'
            df.loc[idx, "previous_docno"] = prev_docno
            df.loc[idx, "previous_docno_str"] = df.iloc[prev_docno]['docno']
            df.loc[idx, "previous_pos"] = df[df.index == prev_docno].original_position.values[0]
        except:
            continue

    # # iterating over groups to get their ranks (done if queries are the normal 31 only)
    # # Grouping the DataFrame by 'round_no' and 'query_id' for further processing
    # df_gb = df.groupby(["query_id_new"])
    #
    # # Looping through each group
    # for group_name, df_group in tqdm(df_gb):
    #     # Splitting the group into 'non_bots' (where 'creator' is "creator") and 'bots' (where 'creator' is not "creator")
    #     if int(str(group_name[0])[0]) not in list(range(int(minimal_epoch), int(maximal_epoch) + 1)):
    #         continue
    #     non_bots = df_group[df_group.creator == "creator"]
    #     bots = df_group[df_group.creator != "creator"]
    #
    #     # Looping through each row in 'bots' group to calculate 'current_pos' for each bot
    #     for idx, row in bots.iterrows():
    #         # Concatenating the 'non_bots' group and the current bot row to create a comparison DataFrame
    #         comp_df = pd.concat([non_bots, row.to_frame().T])
    #
    #         # Removing the current bot from the comparison DataFrame
    #         comp_df = comp_df[(comp_df.username != row.creator) | (comp_df.creator == row.username)]
    #
    #         # Calculating ranks based on the 'score' column in the comparison DataFrame
    #         comp_df['calc_rank'] = comp_df['score'].rank(ascending=False, method='dense')
    #
    #         # Setting the 'current_pos' for the current bot in the main DataFrame
    #         assert comp_df.loc[idx, "calc_rank"] % 1 == 0
    #         df.loc[idx, "current_pos"] = comp_df.loc[idx, "calc_rank"]
    #         if comp_df.loc[idx, "calc_rank"].astype(int) != df.loc[idx, "rank"].astype(int):
    #             x = 1

    df = df.rename({"rank": "current_pos"}, axis=1)

    # df.loc[df.creator == 'creator', 'current_pos'] = df.loc[df.creator == 'creator', "original_position"]

    df['pos_diff'] = df.apply(lambda row: max(int(row['current_pos'] - row['previous_pos']) * -1, 0) if pd.notna(
        row['current_pos']) and pd.notna(row['previous_pos']) else np.nan, axis=1)
    df['scaled_pos_diff'] = df.apply(
        lambda row: row['pos_diff'] / (row['previous_pos'] - 1) if pd.notna(row['previous_pos']) and row[
            'previous_pos'] != 1 else np.nan, axis=1)

    # Calculating 'scaled_orig_pos_diff' based on 'orig_pos_diff', considering some conditions
    df['orig_pos_diff'] = df.apply(
        lambda row: max(int(row['current_pos'] - row['original_position']) * -1, 0) if pd.notna(
            row['current_pos']) and pd.notna(row['original_position']) else np.nan, axis=1)

    df['scaled_orig_pos_diff'] = df.apply(
        lambda row: row['orig_pos_diff'] / (row['original_position'] - 1) if pd.notna(row['original_position']) and row[
            'original_position'] != 1 else np.nan, axis=1)

    df = df[df.round_no != 1]
    df = df[df.score.notna()]

    # TODO: remove this line if condition change - removing docs ranked first
    df = df[df.previous_pos != 1]

    df.to_csv(f"feature_data_{cp}_new.csv", index=False)
    print("Part 2 ended")
    exit()


#### calculate smilarity ####
def get_bert_embeddings(text):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    model = AutoModel.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco").eval()

    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**encoded_input)

    return output.last_hidden_state.mean(dim=1).numpy()


def get_tfidf_embeddings(vectorizer, texts):
    return vectorizer.transform(texts).toarray()


# def get_llama2_embeddings(text):
#     # Initialize the tokenizer
#     global llama_model
#     global llama_tokenizer
#
#     encoded_input = llama_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#
#     with torch.no_grad():
#         output = llama_model(**encoded_input)
#
#     return output.last_hidden_state.mean(dim=1).numpy()

def get_llama2_embeddings(text):
    encoded_input = llama_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = llama_model(**encoded_input)
    return output.last_hidden_state.mean(dim=1).numpy()


def calculate_cosine_similarity(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)[0][0]


def process_row(row, vectorizer):
    prev_df = greg_data[(greg_data.query_id == row.query_id) & (greg_data.round_no == "6")]
    if row.creator == "creator":  # student
        prev_txt = prev_df[prev_df.username == row.username].text.values[0]
    else:  # bot
        prev_txt = prev_df[prev_df.username == row.creator].text.values[0]

    try:
        llama_similarity_prev = calculate_cosine_similarity(get_llama2_embeddings(row['text']),
                                                            get_llama2_embeddings(prev_txt))
    except Exception as e:
        print("ERROR!\n", e)
        x = 1
    # print("Previous LLAMA similarity: ", llama_similarity_prev)

    # bert_similarity = calculate_cosine_similarity(top_bert,get_bert_embeddings(row['text']))
    # print("Top BERT similarity: ", bert_similarity)
    bert_similarity_prev = calculate_cosine_similarity(get_bert_embeddings(row['text']), get_bert_embeddings(prev_txt))
    # print("Previous BERT similarity: ", bert_similarity_prev)

    # tfidf_similarity = calculate_cosine_similarity(top_tfidf,get_tfidf_embeddings(vectorizer, [row['text']]))
    # print("Top TF-IDF similarity: ", tfidf_similarity)
    tfidf_similarity_prev = calculate_cosine_similarity(get_tfidf_embeddings(vectorizer, [row['text']]),
                                                        get_tfidf_embeddings(vectorizer, [prev_txt]))
    # print("Previous TF-IDF similarity: ", tfidf_similarity_prev)

    return row.name, bert_similarity_prev, tfidf_similarity_prev, llama_similarity_prev


if not os.path.exists(f"feature_data_{cp}_new_sim.csv"):
    print("Part 3 started")
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                    use_auth_token='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',
                                                    use_fast=True)

    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                       use_auth_token='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')

    llama_model.to('cpu')
    # TODO: assuming we only work on one epoch!

    orig = pd.read_csv(f"feature_data_{cp}_new.csv")
    df = orig[["round_no", "query_id", "creator", "username"]]
    greg_data = \
        pd.read_csv("greg_data.csv").rename({"position": "original_position", "current_document": "text"}, axis=1)[
            ["round_no", "query_id", "username", "original_position", "text"]]
    bot_followup = pd.read_csv(f"bot_followup_{cp}.csv")
    cols = ['round_no', 'query_id', 'username', 'text']
    # prev_top_data = greg_data[
    #     (greg_data.original_position == 1) & (greg_data.round_no.isin([e - 1 for e in df.round_no.unique()]))]

    for col in ["round_no", "query_id", "creator", "username"]:
        df[col] = df[col].astype(str)
        bot_followup[col] = bot_followup[col].astype(str)
        if col != 'creator':
            greg_data[col] = greg_data[col].astype(str)

    bots_df = pd.merge(left=df[df.creator != "creator"], right=bot_followup, on=['query_id', 'creator', 'username'],
                       how='left')
    non_bots_df = pd.merge(left=df[df.creator == "creator"].drop_duplicates(),
                           right=greg_data[['query_id', 'username', 'text']], on=['query_id', 'username'],
                           how='left').rename({"round_no_x": "round_no"})
    texts_df = pd.concat([bots_df[["query_id", "creator", "username", "text"]],
                          non_bots_df[["query_id", "creator", "username", "text"]]]).sort_values("query_id")
    texts_df["top_bert_sim"], texts_df["top_tfidf_sim"] = np.nan, np.nan

    vectorizer = TfidfVectorizer()
    # vectorizer.fit(texts_df.text.values.tolist() + prev_top_data.text.values.tolist())
    vectorizer.fit(texts_df.text.values.tolist())
    # for qid in list(texts_df.query_id.unique()):
    #     top_txt = prev_top_data[prev_top_data.query_id == int(qid)].text.values[0]
    #     top_bert = get_bert_embeddings(top_txt)
    #     top_tfidf = get_tfidf_embeddings(vectorizer,[top_txt])
    #     for index, row in tqdm(texts_df[texts_df.query_id == qid].iterrows()):
    #         # Calculate BERT similarity
    #         bert_similarity = calculate_cosine_similarity(
    #             top_bert,
    #             get_bert_embeddings(row['text'])
    #         )
    #         texts_df.at[index, "top_bert_sim"] = bert_similarity
    #
    #         # Calculate TF-IDF similarity
    #         tfidf_similarity = calculate_cosine_similarity(
    #             top_tfidf,
    #             get_tfidf_embeddings(vectorizer,[row['text']])
    #         )
    #         texts_df.at[index, "top_tfidf_sim"] = tfidf_similarity
    #
    #         x = 1
    #     texts_df.to_csv(f"feature_data_{cp}_new_sim.csv", index = False)
    # x = 1
    # Prepare ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=12)

    # Collect all futures
    futures = []

    for qid in tqdm(list(texts_df.query_id.unique()), desc="Processing queries", miniters=100,
                    total=len(list(texts_df.query_id.unique()))):
        # top_txt = prev_top_data[prev_top_data.query_id == int(qid)].text.values[0]
        # top_bert = get_bert_embeddings(top_txt)
        # top_tfidf = get_tfidf_embeddings(vectorizer, [top_txt])

        subset_df = texts_df[texts_df.query_id == qid]
        for index, row in subset_df.iterrows():
            futures.append(executor.submit(process_row, row, vectorizer))

    # Retrieve results and update DataFrame
    for future in tqdm(futures, desc="Updating DataFrame", miniters=100, total=len(futures)):
        index, prev_bert_sim, prev_tfidf_sim, prev_llama_sim = future.result()
        # texts_df.at[index, "top_bert_sim"] = top_bert_sim
        # texts_df.at[index, "top_tfidf_sim"] = top_tfidf_sim
        texts_df.at[index, "prev_bert_sim"] = prev_bert_sim
        texts_df.at[index, "prev_tfidf_sim"] = prev_tfidf_sim
        texts_df.at[index, "prev_llama_sim"] = prev_llama_sim

    final = texts_df.merge(orig, on=["query_id", "creator", "username"], how="left")
    final.to_csv(f"feature_data_{cp}_new_sim.csv", index=False)

    # Clean up
    executor.shutdown()
    print("Part 3 ended")
