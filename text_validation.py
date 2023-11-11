import pandas as pd
from tqdm import tqdm
from config import current_prompt as cp, bot_cover_df
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")


def read_query_xml_to_dict(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    queries_dict = {}

    for query in root.findall('query'):
        number = query.find('number').text
        text_element = query.find('text').text

        prefix = "#combine("
        suffix = ")"
        if text_element.startswith(prefix) and text_element.endswith(suffix):
            text = text_element[len(prefix):-len(suffix)].strip()
        else:
            text = text_element
        queries_dict[int(number)] = text
    return queries_dict


def create_query_xml(queries_dict, filename):
    # Create the root element
    parameters = ET.Element('parameters')

    # Iterate over the dictionary and create query elements
    for query_id, query_text in queries_dict.items():
        # Create a query element
        query_elem = ET.SubElement(parameters, 'query')
        # Create a number element and set its text
        number_elem = ET.SubElement(query_elem, 'number')
        number_elem.text = str(query_id).zfill(3)  # Ensure the id is in the format 'XXX'
        # Create a text element and set its text
        text_elem = ET.SubElement(query_elem, 'text')
        text_elem.text = f"#combine( {query_text} )"

    # Create an ElementTree object and write to the file
    tree = ET.ElementTree(parameters)
    ET.indent(tree, space="  ", level=0)  # Properly format the XML for pretty printing
    tree.write(filename, encoding='utf-8', xml_declaration=True)


texts = pd.read_csv(f"bot_followup_{cp}.csv").sort_values(['query_id', 'round_no']).dropna()
# fix bot docno rounds
for idx, row in texts[texts.creator != 'creator'].iterrows():
    texts.at[idx, 'round_no'] = row.round_no + 1

rounds = texts.round_no.unique()
greg_data = pd.read_csv("greg_data.csv").rename({"current_document": "text"}, axis=1)
greg_data["creator"] = "creator"
names = greg_data[["query_id", "query"]].set_index('query_id').to_dict()['query']
greg_data = greg_data[texts.columns]
greg_data = greg_data[greg_data.round_no.isin(rounds)]

# asrc_df = pd.read_csv("bot_followup_asrc.csv")
# df = pd.concat([greg_data, texts, asrc_df])
df = pd.concat([greg_data, texts]).sort_values(['round_no', 'query_id'])
# df = df[df.round_no.isin(rounds_comp)]
df["docno"] = df.apply(lambda row: "{}-{}-{}-{}".format('0' + str(row.round_no),
                                                        '0' + str(row.query_id) if row.query_id < 100 else str(
                                                            row.query_id), row.username, row.creator), axis=1)

df = df[df.round_no != 1]
df = df.dropna().sort_values(['round_no', 'query_id', 'username']).reset_index(drop=True)
df = pd.merge(df, bot_cover_df.reset_index()[["index","bot_name"]], left_on='username', right_on='bot_name', how='left').drop("bot_name", axis=1)
x = 1
# long_texts = []
# short_texts = []
# for idx, row in tqdm(df.iterrows()):
#     if "BOT" not in str(row.username):
#         continue
#
#     if len(row.text.split(' ')) < 140:
#         # print("SHORT TEXT\n")
#         # print(
#         #     f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, creator: {row.creator}, username: {row.username}, length: {len(row.text.split(' '))}\n")
#         short_texts.append(idx + 2)
#
#     if len(row.text.split(' ')) > 150:
#         # print("LONG TEXT! CHANGE!\n")
#         # print(
#         #     f"idx in file: {idx + 2}, id: {row.query_id}, topic: {names[row.query_id]}, creator: {row.creator}, username: {row.username}, length: {len(row.text.split(' '))}\n")
#         long_texts.append(idx + 2)
#
# print("long texts:", sorted(long_texts))
# print("short texts:", sorted(short_texts))

# create trectext format

# create working set file
# 201 Q0 ROUND-01-002-44 0 2.1307966709136963 summarizarion_task
# 201 Q0 ROUND-01-002-14 0 -0.1451454907655716 summarizarion_task
# 201 Q0 ROUND-01-002-29 0 -1.9667246341705322 summarizarion_task
# 201 Q0 ROUND-01-002-13 0 -3.396240472793579 summarizarion_task


working_set_docnos = 0
gb_df = df.groupby(["round_no", "query_id"])
query_dict = read_query_xml_to_dict('/lv_local/home/niv.b/content_modification_code-master/data/queries_bot_modified_sorted_1.xml')
new_query_dict = {}
comp_dict = {}
for group_name, df_group in tqdm(gb_df):
    creators = df_group[df_group.creator != "creator"].creator.unique()
    bots = df_group[df_group.creator != "creator"].username.unique()

    for creator in creators:
        for bot in bots:
            comp_df = df_group[((df_group.username != creator) & (df_group.creator == "creator")) | (
                        (df_group.username == bot) & (df_group.creator == creator))]
            ind = int(comp_df[comp_df.username == bot]["index"].iloc[0])
            key = str(group_name[0]) + str(group_name[1]).rjust(3, '0') + str(creator).rjust(2, '0') + str(ind).rjust(3, '0')
            comp_df.loc[:, 'Key'] = key
            comp_dict[key] = comp_df
            new_query_dict[key] = query_dict[group_name[1]]

create_query_xml(new_query_dict, f'/lv_local/home/niv.b/content_modification_code-master/data/query_files/queries_{cp}.xml')

result = pd.concat(comp_dict.values(), axis=0)

with open(f"working_set_{cp}.trectext", "w") as f:
    for idx, row in result.sort_values("Key").iterrows():
        f.write(f"{row.Key} Q0 {row.docno} 0 1.0 summarizarion_task\n")
        working_set_docnos += 1

bot_followup_docnos = 0
with open(f"bot_followup_{cp}.trectext", "w") as f:
    f.write(f"<DATA>\n")
    for idx, row in df.iterrows():
        f.write(f"<DOC>\n")
        f.write(f"<DOCNO>{row.docno}</DOCNO>\n")
        f.write(f"<TEXT>\n")
        f.write(f"{row.text}\n")
        f.write(f"</TEXT>\n")
        f.write(f"</DOC>\n")
        bot_followup_docnos += 1
    f.write(f"</DATA>\n")

x = 1
