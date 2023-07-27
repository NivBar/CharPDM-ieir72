import pandas as pd
from pprint import pprint
from config import current_prompt as cp

texts = pd.read_csv(f"bot_followup_{cp}.csv").sort_values('query_id')
texts["round_no"] = 4
greg_data = pd.read_csv("greg_data.csv").rename({"TEXT": "text"}, axis=1)
# names = pd.read_csv("competition-topics.tsv", sep="\t").set_index('TRECTopicNumber').to_dict()['TRECQuery']
names = greg_data[["query_id", "query"]].set_index('query_id').to_dict()['query']
greg_data = greg_data[['group', 'query_id', 'username', 'text', 'round_no']]

df = pd.concat([greg_data, texts])
df["docno"] = df.apply(lambda row: "ROUND-{}-{}-{}".format('0' + str(row.round_no),
                                                           '0' + str(row.query_id) if row.query_id < 100 else str(
                                                               row.query_id), row.username), axis=1)

df = df.dropna()
long_texts = []
short_texts = []
for idx, row in texts.iterrows():
    if len(row.text.split(' ')) < 140:
        print("SHORT TEXT\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        short_texts.append(idx + 2)

    if len(row.text.split(' ')) > 150:
        print("LONG TEXT! CHANGE!\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        long_texts.append(idx + 2)

    if "rank" in row.text:
        print("RANK IN TEXT\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
        print(f"{row.text}\n\n")
    if "LOOP BREAK" in row.text:
        print("LOOP BREAK IN TEXT\n")
        print(
            f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")

    # print(f"idx in file: {idx+2}, id: {row.query_id}, topic: {names[row.query_id]}, group: {row.group}, username: {row.username}, length: {len(row.text.split(' '))}\n")
    # print(f"{row.text}\n\n")
print("long texts:", sorted(long_texts))
print("short texts:", sorted(short_texts))

# create trectext format
with open(f"bot_followup_{cp}.trectext", "w") as f:
    f.write(f"<DATA>\n")
    for idx, row in df.iterrows():
        f.write(f"<DOC>\n")
        f.write(f"<DOCNO>{row.docno}</DOCNO>\n")
        f.write(f"<TEXT>\n")
        f.write(f"{row.text}\n")
        f.write(f"</TEXT>\n")
        f.write(f"</DOC>\n")
    f.write(f"</DATA>\n")

# create working set file
# 201 Q0 ROUND-01-002-44 0 2.1307966709136963 summarizarion_task
# 201 Q0 ROUND-01-002-14 0 -0.1451454907655716 summarizarion_task
# 201 Q0 ROUND-01-002-29 0 -1.9667246341705322 summarizarion_task
# 201 Q0 ROUND-01-002-13 0 -3.396240472793579 summarizarion_task
with open(f"working_set_{cp}.trectext", "w") as f:
    for idx, row in df[df.round_no == 4].sort_values(["query_id", "username"]).iterrows():
        f.write(
            f"{'0' + str(row.query_id) if row.query_id < 100 else str(row.query_id)} Q0 {row.docno} 0 1.0 summarizarion_task\n")
x = 1
