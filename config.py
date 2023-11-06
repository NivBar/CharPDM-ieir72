import random
from pprint import pprint

import openai
import pandas as pd
from API_key import API_key

current_prompt = "asrc"

#### bot names ####
# def get_names_dict(markov=False):
#     if markov:
#         return {"all": "MABOT", "tops": "MTBOT", "self": "MSBOT"}
#     else:
#         return {"all": "NMABOT", "tops": "NMTBOT", "self": "NMSBOT"}
bot_cover_df = pd.read_csv("prompt_options.csv")
ACTIVE_BOTS = bot_cover_df.bot_name.tolist()

#### openai parameters ####
openai.api_key = API_key

"""
Model: Determines the architecture and parameters of the language model used for text generation. Different models have 
different strengths and weaknesses for specific types of text generation tasks.

Temperature: Controls the level of randomness and creativity in the generated text. High temperature values (e.g., 1.0 
or higher) can produce more diverse and unexpected outputs, while low values (e.g., 0.5 or lower) can produce more 
predictable and conservative outputs.

Top_p: Limits the set of possible next words based on the model's predictions. High top_p values (e.g., 0.9 or higher) 
allow for more variation and creativity, while low values (e.g., 0.1 or lower) can produce more predictable and 
conservative outputs.

Max_tokens: Sets an upper limit on the number of tokens that can be generated in the output text. High max_tokens values 
(e.g., 500 or higher) can produce longer outputs, while low values (e.g., 50 or lower) can produce shorter and more 
concise outputs.

Frequency_penalty: Encourages the model to generate less frequent words or phrases. High frequency_penalty values 
(e.g., 2.0 or higher) can increase the diversity and creativity of the generated text, while low values 
(e.g., 0.5 or lower) can produce more common and predictable outputs.

Presence_penalty: Encourages the model to avoid repeating words or phrases that have already appeared in the output 
text. High presence_penalty values (e.g., 2.0 or higher) can promote the generation of novel and varied text, while low 
values (e.g., 0.5 or lower) can produce more repetitive and redundant outputs.
"""
model = "gpt-4"
temperature = 0.2
top_p = 0.3
max_tokens = 250
frequency_penalty = 1.0
presence_penalty = 0.0

#### useful data collections ####
# topic_codex_new = json.load(open("topic_queries_doc.json", "r"))
# topic_codex = dict()

# TODO: change to actual copetition data when starting
comp_data = pd.read_csv("Archive/comp_dataset.csv")

query_index = {x[0]: x[1] for x in comp_data[["query_id", "query"]].drop_duplicates().values.tolist()}


def get_prompt(bot_name, data, creator_name, query_id):
    method, traits = bot_name.split(
        "_")  # bot names - {method}_[query no}{example no}{candidate inc}{query inc}{doc type}{history len}
    bot_info = {"method": method, "query_num": int(traits[0]), "ex_num": int(traits[1]),
                "cand_inc": True if traits[2] == "1" else False,
                "query_inc": True if traits[3] == "1" else False}
    if bot_info["method"] in ["DYN", "PAW"]:
        bot_info["doc_type"] = traits[4]
        if bot_info["method"] == "DYN":
            bot_info["history_len"] = int(traits[5])

    print(bot_info)

    query_ids = random.sample(data[data.query_id != query_id]["query_id"].unique().tolist(), bot_info["query_num"])
    if bot_info["query_inc"]:
        query_ids[0] = query_id

    recent_data = data[data['query_id'] == query_id]
    query_string = recent_data.iloc[0]['query']
    epoch = int(max(recent_data["round_no"]))
    current_doc = \
        recent_data[(recent_data.round_no == epoch) & (recent_data.username == creator_name)][
            "current_document"].values[0].strip()
    current_rank = \
        recent_data[(recent_data.round_no == epoch) & (recent_data.username == creator_name)]["position"].values[0]
    previos_doc = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == creator_name)][
        "current_document"].values[0]
    previous_rank = \
        recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == creator_name)]["position"].values[0]
    tops = recent_data[recent_data["position"] == int(min(recent_data["position"]))]
    tops_docs = "\n\n".join(tops["current_document"].values)
    bottoms = recent_data[recent_data["position"] == int(max(recent_data["position"]))]
    bottoms_docs = "\n\n".join(bottoms["current_document"].values)
    top_doc_txt = tops[tops["round_no"] == epoch]["current_document"].values[0]
    top_doc_user = tops[tops["round_no"] == epoch]["username"].values[0]
    top_prev_doc = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == top_doc_user)][
        "current_document"].values[0]
    top_prev_rank = \
        recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == top_doc_user)]["position"].values[0]
    bottom_doc_txt = bottoms[bottoms["round_no"] == epoch]["current_document"].values[0]
    bottom_doc_user = bottoms[bottoms["round_no"] == epoch]["username"].values[0]
    bottom_prev_doc = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == bottom_doc_user)][
        "current_document"].values[0]
    bottom_prev_rank = \
        recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == bottom_doc_user)]["position"].values[
            0]

    if bot_info["cand_inc"]:
        message = [{"role": "system",
                    "content": fr"The candidate document, ranked {current_rank} in round {epoch} is:\n {current_doc}"}]
        all_docs = "\n".join(f"{row['position']}. {row['current_document'].strip()}" for _, row in recent_data[
            (recent_data.round_no == epoch) & (recent_data.username != creator_name)].
                             sort_values("position").iterrows())
    else:
        message = [{"role": "system",
                    "content": fr"The candidate document is: {current_doc}"}]
        all_docs = "\n".join(f"* {doc.strip()}" for doc in recent_data[(recent_data.round_no == epoch) &
                                                                       (
                                                                               recent_data.username != creator_name)].sort_values(
            "position")["current_document"].values)

    if bot_info["method"] == "POW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            message_list = [f'Document ranked 1 in epoch {epoch} in relation to the query mentioned:\n {top_doc_txt}']

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0]
                    message_list.append(f'Document ranked 1 in epoch {epoch - i} in relation to the query mentioned:\n'
                                        f' {top_doc_txt}')
            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                message_list.append(f'In the case of observing the query - {query_string}:\n')
                tops = data[(data["position"] == int(min(data["position"]))) & (data["query_id"] == qid)]

                for i in range(bot_info["ex_num"]):
                    top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0]
                    message_list.append(f'Document ranked 1 in epoch {epoch - i}:\n {top_doc_txt}')
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    elif bot_info["method"] == "PAW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            if bot_info["doc_type"] == "T":
                top_doc_txt = tops[tops["round_no"] == epoch]["current_document"].values[0]
                if bot_info["cand_inc"]:
                    message_list = [
                        f'This is the candidate document\'s competitor ranked 1 in epoch {epoch} in relation to the query '
                        f'mentioned:\n{top_doc_txt}\n']
                else:
                    rand_doc_row = data[
                        (data["round_no"] == epoch) & (data.username != creator_name) & (data.position != 1) & (
                                    data.query_id == query_id)].sample(n=1)
                    rand_doc_txt = rand_doc_row["current_document"].values[0]
                    rand_doc_pos = rand_doc_row["position"].values[0]
                    message_list = [f'Documents ranked 1 and {rand_doc_pos} respectively in epoch {epoch} '
                                    f'in relation to the query mentioned:\n1. {top_doc_txt}\n\n{rand_doc_pos}. '
                                    f'{rand_doc_txt}\n']
            else:
                rand_doc_row = data[
                    (data["round_no"] == epoch) & (data.username != creator_name) & (data.query_id == query_id)].sample(
                    n=1)
                rand_doc_txt = rand_doc_row["current_document"].values[0]
                rand_doc_pos = rand_doc_row["position"].values[0]
                message_list = [
                    f'This is the candidate document\'s competitor ranked {rand_doc_pos} in epoch {epoch} in relation to the query '
                    f'mentioned:\n{rand_doc_txt}\n']

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    if bot_info["doc_type"] == "T":
                        top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0].strip()
                        rand_doc_row = data[(data["round_no"] == epoch - i) & (data.position != 1) & (
                                    data.query_id == query_id)].sample(n=1)
                        rand_doc_txt = rand_doc_row["current_document"].values[0].strip()
                        rand_doc_pos = rand_doc_row["position"].values[0]
                        message_list.append(f'Documents ranked 1 and {rand_doc_pos} respectively in epoch {epoch - i} '
                                            f'in relation to the query mentioned:\n1. {top_doc_txt}\n\n{rand_doc_pos}. '
                                            f'{rand_doc_txt}\n')
                    else:
                        rand_doc_rows = data[(data["round_no"] == epoch - i) & (data.username != creator_name) &
                                             (data.query_id == query_id)].sample(n=2, replace=False).sort_values(
                            "position")
                        rand_pos = rand_doc_rows['position'].values
                        rand_docs = rand_doc_rows['current_document'].values
                        message_list.append(
                            f'Documents ranked {rand_pos[0]} and {rand_pos[1]} respectively in epoch {epoch - i} '
                            f'in relation to the query mentioned:\n{rand_pos[0]}. {rand_docs[0].strip()}\n\n{rand_pos[1]}. '
                            f'{rand_docs[1].strip()}\n')

            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                message_list.append(f'In the case of observing the query - {query_string}:\n')

                for i in range(bot_info["ex_num"]):
                    if bot_info["doc_type"] == "T":
                        tops = data[(data["position"] == int(min(recent_data["position"]))) & (data.query_id == qid)]
                        top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0].strip()
                        rand_doc_row = data[(data["round_no"] == epoch - i) & (data.position != 1) & (
                                data.query_id == qid)].sample(n=1)
                        rand_doc_txt = rand_doc_row["current_document"].values[0].strip()
                        rand_doc_pos = rand_doc_row["position"].values[0]
                        message_list.append(
                            f'Documents ranked 1 and {rand_doc_pos} respectively in epoch {epoch - i}: \n'
                            f'1. {top_doc_txt}\n\n{rand_doc_pos}. {rand_doc_txt}\n')
                    else:
                        rand_doc_rows = data[(data["round_no"] == epoch - i) & (data.username != creator_name) &
                                             (data.query_id == qid)].sample(n=2, replace=False).sort_values(
                            "position")
                        rand_pos = rand_doc_rows['position'].values
                        rand_docs = rand_doc_rows['current_document'].values
                        message_list.append(
                            f'Documents ranked {rand_pos[0]} and {rand_pos[1]} respectively in epoch {epoch - i}: \n'
                            f'{rand_pos[0]}. {rand_docs[0].strip()}\n\n{rand_pos[1]}. {rand_docs[1].strip()}\n')
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    elif bot_info["method"] == "LIW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            message_list = [
                f'These are the candidate document\'s ranked competitors, ordered from highst to lowest, in '
                f'epoch {epoch} in relation to the query mentioned:\n{all_docs}\n']

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    all_docs = "\n".join(f"{i + 1}. {doc.strip()}" for i, doc in enumerate(
                        recent_data[recent_data.round_no == epoch - i].sort_values("position")
                        ["current_document"].values))
                    message_list.append(
                        f'Ranked documents, ordered from first to last, in epoch {epoch - i}:\n{all_docs}\n')
            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                message_list.append(f'In the case of observing the query - {query_string}:\n')

                for i in range(bot_info["ex_num"]):
                    all_docs = "\n".join(f"{i + 1}. {doc.strip()}" for i,
                                                                       doc in
                                         enumerate(data[(data.round_no == epoch - i) &
                                                        (data['query_id'] == qid)].sort_values("position")[
                                                       "current_document"].values))
                    message_list.append(
                        f'Ranked documents, ordered from first to last, in epoch {epoch - i}:\n{all_docs}\n')
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    elif bot_info["method"] == "DYN":
        epochs = [epoch - i for i in range(bot_info["history_len"])]
        strands = bot_info["ex_num"]
        if query_ids[0] == query_id:
            message_list = []
            query_ids.pop(0)
            # top doc strand
            if bot_info["doc_type"] == "T":
                strands -= 1
                cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username == top_doc_user) &
                                     (data.query_id == query_id)].sort_values("round_no", ascending=False)
                cand_pos = cand_doc_rows['position'].values
                cand_docs = cand_doc_rows['current_document'].values
                cand_eps = cand_doc_rows['round_no'].values
                message_string = f'The ranking history of the user that created the top document in relation to the ' \
                                 f'query mentioned is as follows:\n'
                for i in range(len(cand_eps)):
                    message_string += f'In epoch {cand_eps[i]}, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                message_list.append(message_string)

            # candidate doc strand
            if bot_info["cand_inc"] and current_rank != 1 and strands > 0:
                strands -= 1
                cand_doc_rows = data[(data["round_no"].isin(epochs[1:])) & (data.username == creator_name) &
                                     (data.query_id == query_id)].sort_values("round_no", ascending=False)
                cand_pos = cand_doc_rows['position'].values
                cand_docs = cand_doc_rows['current_document'].values
                cand_eps = cand_doc_rows['round_no'].values
                message_string = f'The ranking history of the user that created the candidate document is as follows:\n'
                for i in range(len(cand_eps)):
                    message_string += f'In epoch {cand_eps[i]}, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                message_list.append(message_string)


            if strands > 0:
                cand_users = data[(data["round_no"].isin(epochs)) & ~(data.username.isin([top_doc_user, creator_name]))
                                    & (data.query_id == query_id)]['username'].sample(n=strands, replace=False).tolist()
                cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username.isin(cand_users))
                                    & (data.query_id == query_id)].sort_values("round_no", ascending=False)


                for user in cand_users:
                    user_data = cand_doc_rows[cand_doc_rows.username == user]
                    cand_pos = user_data['position'].values
                    cand_docs = user_data['current_document'].values
                    cand_eps = user_data['round_no'].values
                    message_string = f'The ranking history of the user that created the document ranked {cand_pos[0]} in relation to the ' \
                                     f'query mentioned is as follows:\n'
                    for i in range(len(cand_eps)):
                        message_string += f'In epoch {cand_eps[i]}, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                    message_list.append(message_string)

            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                strands = bot_info["ex_num"]
                query_string = data[data['query_id'] == qid].iloc[0]['query']
                message_list = [f'In the case of observing the query - {query_string}:\n']

                if bot_info["doc_type"] == "T":
                    strands -= 1
                    top_doc_user = data[(data['query_id'] == qid) & (data["position"] == 1) & (data["round_no"] == epoch)]['username'].values[0]

                    cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username == top_doc_user) &
                                             (data.query_id == qid)].sort_values("round_no", ascending=False)
                    cand_pos = cand_doc_rows['position'].values
                    cand_docs = cand_doc_rows['current_document'].values
                    cand_eps = cand_doc_rows['round_no'].values
                    message_string = f'The ranking history of the user that created the top document is as follows:\n'
                    for i in range(len(cand_eps)):
                        message_string += f'In epoch {cand_eps[i]}, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                    message_list.append(message_string)

                if strands > 0:
                    cand_users = \
                    data[(data["round_no"].isin(epochs)) & ~(data.username.isin([top_doc_user, creator_name]))
                         & (data.query_id == qid)]['username'].sample(n=strands, replace=False).tolist()
                    cand_doc_rows = data[(data["round_no"].isin(epochs)) & (data.username.isin(cand_users))
                                         & (data.query_id == qid)].sort_values("round_no", ascending=False)

                    for user in cand_users:
                        user_data = cand_doc_rows[cand_doc_rows.username == user]
                        cand_pos = user_data['position'].values
                        cand_docs = user_data['current_document'].values
                        cand_eps = user_data['round_no'].values
                        message_string = f'The ranking history of the user that created the document ranked {cand_pos[0]} is as follows:\n'
                        for i in range(len(cand_eps)):
                            message_string += f'In epoch {cand_eps[i]}, ranked {cand_pos[i]}:\n{cand_docs[i].strip()}\n'
                        message_list.append(message_string)
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    # PROMPT_BANK = {
    #     "INFOBOT1":
    #         {'role': 'user', 'content': f'This is the document ranked first in epoch {epoch}:\n {top_doc_txt}.\n'
    #                                     f'Analyze this text to understand why it is ranked first and edit the candidate '
    #                                     f'document to ensure the edited document ranks first in the next epoch.'},
    #     "INFOBOT2":
    #         {'role': 'user', 'content': f'This is the document ranked last in epoch {epoch}:\n {bottom_doc_txt}.\n'
    #                                     f'Analyze this text to understand why it is ranked last and edit the candidate '
    #                                     f'document to ensure the edited document ranks first in the next epoch.'},
    #     "INFOBOT3":
    #         {'role': 'user', 'content': f'These are the documents ranked first and last in epoch {epoch}:\n'
    #                                     f'First ranked document:\n {top_doc_txt}.\n Last ranked document:\n {bottom_doc_txt}.\n'
    #                                     f'Analyze these texts to understand why they were ranked the way they did and '
    #                                     f'edit the candidate document to ensure the edited document ranks first in the next epoch.'},
    #     "INFOBOT4":
    #         {'role': 'user',
    #          'content': f'This is your document ranked {previous_rank} in epoch {epoch - 1}:\n {previos_doc}.\n'
    #                     f'Analyze these texts to understand the rank changes if occured and edit the candidate '
    #                     f'document to ensure the edited document ranks first in the next epoch.'},
    #     "INFOBOT5":
    #         {'role': 'user',
    #          'content': f'These are the ranked documents, ordered from first to last, in epoch {epoch}:\n'
    #                     f'{all_docs}.\n'
    #                     f'Analyze these texts to understand why they were ranked the way they did and '
    #                     f'edit the candidate document to ensure the edited document ranks first in the next epoch.'},
    #     "OBLIBOT1":
    #         {'role': 'user', 'content': f'This is the document ranked first in epoch {epoch}:\n {top_doc_txt}.\n'
    #                                     f'And this is the same user\'s document ranked {top_prev_rank} in epoch {epoch - 1}:\n {top_prev_doc}.\n'
    #                                     f'Analyze these texts to understand the rank changes if occured and edit the candidate '
    #                                     f'document to ensure the edited document ranks first in the next epoch.'},
    #     "OBLIBOT2":
    #         {'role': 'user', 'content': f'This is the document ranked last in epoch {epoch}: {bottom_doc_txt}.\n'
    #                                     f'And this is the same user\'s document ranked {bottom_prev_rank} in epoch {epoch - 1}: {bottom_prev_doc}.\n'
    #                                     f'Analyze these texts to understand the rank changes if occured and edit the candidate '
    #                                     f'document to ensure the edited document ranks first in the next epoch.'},
    #     "OBLIBOT3":
    #         {'role': 'user', 'content': f'These are the top documents from the last 3 epochs:\n'
    #                                     f'Top docs:\n {tops_docs}\n'
    #                                     f'Analyze these texts to understand why they were ranked the way they did and '
    #                                     f'edit the candidate document to ensure the edited document ranks first in the next epoch.'},
    #     "OBLIBOT4":
    #         {'role': 'user', 'content': f'These are the bottom documents from the last 3 epochs:\n'
    #                                     f'Bottom docs:\n {bottoms_docs}\n'
    #                                     f'Analyze these texts to understand why they were ranked the way they did and '
    #                                     f'edit the candidate document to ensure the edited document ranks first in the next epoch.'},
    #     "OBLIBOT5":
    #         {'role': 'user', 'content': f'These are the top and bottom documents from the last 3 epochs:\n'
    #                                     f'Bottom docs:\n {bottoms_docs}\n'
    #                                     f'Top docs:\n {tops_docs}\n'
    #                                     f'Analyze these texts to understand why they were ranked the way they did and '
    #                                     f'edit the candidate document to ensure the edited document ranks first in the next epoch.'}
    # }
    # message.append(PROMPT_BANK[bot_name])
    pprint(message)
    return message


if __name__ == '__main__':
    data = pd.read_csv("sandbox_data.csv")
    get_prompt("DYN_3311R3", data, 51, 195)
