import pandas as pd
# from pprint import pprint
from config import current_prompt as cp
from tqdm import tqdm

texts = pd.read_csv(f"bot_followup_{cp}.csv").sort_values(['query_id','round_no'])
# texts["round_no"] = 4
greg_data = pd.read_csv("greg_data.csv").rename({"current_document": "text"}, axis=1)
greg_data["creator"] = "creator"
names = greg_data[["query_id", "query"]].set_index('query_id').to_dict()['query']
greg_data = greg_data[texts.columns]

df = pd.concat([greg_data, texts])
df["docno"] = df.apply(lambda row: "{}-{}-{}-{}".format('0' + str(row.round_no),'0' + str(row.query_id) if row.query_id < 100 else str(row.query_id), row.username, row.creator), axis=1)

df = df.dropna()
long_texts = []
short_texts = []
for idx, row in tqdm(texts.iterrows()):
    if len(row.text.split(' ')) < 140:
        print("SHORT TEXT\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, creator: {row.creator}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        short_texts.append(idx + 2)

    if len(row.text.split(' ')) > 150:
        print("LONG TEXT! CHANGE!\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, creator: {row.creator}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        long_texts.append(idx + 2)

print("long texts:", sorted(long_texts))
print("short texts:", sorted(short_texts))

# create trectext format
bot_followup_docnos = 0
with open(f"bot_followup_{cp}.trectext", "w") as f:
    f.write(f"<DATA>\n")
    for idx, row in tqdm(df.iterrows()):
        f.write(f"<DOC>\n")
        f.write(f"<DOCNO>{row.docno}</DOCNO>\n")
        f.write(f"<TEXT>\n")
        f.write(f"{row.text}\n")
        f.write(f"</TEXT>\n")
        f.write(f"</DOC>\n")
        bot_followup_docnos += 1
    f.write(f"</DATA>\n")

# create working set file
# 201 Q0 ROUND-01-002-44 0 2.1307966709136963 summarizarion_task
# 201 Q0 ROUND-01-002-14 0 -0.1451454907655716 summarizarion_task
# 201 Q0 ROUND-01-002-29 0 -1.9667246341705322 summarizarion_task
# 201 Q0 ROUND-01-002-13 0 -3.396240472793579 summarizarion_task
working_set_docnos = 0
with open(f"working_set_{cp}.trectext", "w") as f:
    for idx, row in tqdm(df.sort_values(["query_id", "username"]).iterrows()):
        f.write(
            f"{'0' + str(row.query_id) if row.query_id < 100 else str(row.query_id)} Q0 {row.docno} 0 1.0 summarizarion_task\n")
        working_set_docnos += 1
x = 1
