import random

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
# model = "gpt-3.5-turbo"
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
    method, traits = bot_name.split("_")  # bot names - {method}_[query no}{example no}{candidate inc}{query inc}{doc type}{history len}
    bot_info = {"method": method, "query_num": int(traits[0]), "ex_num": int(traits[1]), "cand_inc": True if traits[2] == "1" else False,
                "query_inc": True if traits[3] == "1" else False}
    if bot_info["method"] in ["DYN", "PAW"]:
        bot_info["doc_type"] = traits[4]
        if bot_info["method"] == "DYN":
            bot_info["history_len"] = int(traits[5])

    query_ids = random.sample(data[data.query_id != query_id]["query_id"].unique().tolist(),bot_info["query_num"])
    if bot_info["query_inc"]:
        query_ids[0] = query_id

    recent_data = data[data['query_id'] == query_id]
    query_string = recent_data.iloc[0]['query']
    epoch = int(max(recent_data["round_no"]))
    current_doc = recent_data[(recent_data.round_no == epoch) & (recent_data.username == creator_name)]["current_document"].values[0]
    current_rank = recent_data[(recent_data.round_no == epoch) & (recent_data.username == creator_name)]["position"].values[0]
    previos_doc = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == creator_name)]["current_document"].values[0]
    previous_rank = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == creator_name)]["position"].values[0]
    tops = recent_data[recent_data["position"] == int(min(recent_data["position"]))]
    tops_docs = "\n\n".join(tops["current_document"].values)
    bottoms = recent_data[recent_data["position"] == int(max(recent_data["position"]))]
    bottoms_docs = "\n\n".join(bottoms["current_document"].values)
    top_doc_txt = tops[tops["round_no"] == epoch]["current_document"].values[0]
    top_doc_user = tops[tops["round_no"] == epoch]["username"].values[0]
    top_prev_doc = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == top_doc_user)]["current_document"].values[0]
    top_prev_rank = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == top_doc_user)]["position"].values[0]
    bottom_doc_txt = bottoms[bottoms["round_no"] == epoch]["current_document"].values[0]
    bottom_doc_user = bottoms[bottoms["round_no"] == epoch]["username"].values[0]
    bottom_prev_doc = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == bottom_doc_user)]["current_document"].values[0]
    bottom_prev_rank = recent_data[(recent_data.round_no == epoch - 1) & (recent_data.username == bottom_doc_user)]["position"].values[0]
    all_docs = "\n\n".join(recent_data[recent_data.round_no == epoch].sort_values("position")["current_document"].values)

    if bot_info["cand_inc"]:
        message = [{"role": "system",
                    "content": fr"The candidate document, ranked {current_rank} in round {epoch} is:\n {current_doc}"}]
    else:
        message = [{"role": "system",
                    "content": fr"The candidate document is: {current_doc}"}]

    if bot_info["method"] == "POW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            message_list = [f'Document ranked 1 in epoch {epoch} in relation to the query mentioned:\n {top_doc_txt}']

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0]
                    message_list.append(f'Document ranked 1 in epoch {epoch - i} in relation to the query mentioned:\n {top_doc_txt}')
            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = tops.iloc[0]['query']
                message_list.append(f'In the case of observing the query - {query_string}:\n')
                tops = data[(data["position"] == int(min(data["position"]))) & (data["query_id"] == qid)]

                for i in range(bot_info["ex_num"]):
                    top_doc_txt = tops[tops["round_no"] == epoch - i]["current_document"].values[0]
                    message_list.append(f'Document ranked 1 in epoch {epoch - i}:\n {top_doc_txt}')
                message.append({'role': 'user', 'content': '\n'.join(message_list)})


    elif bot_info["method"] == "PAW":
        pass

    elif bot_info["method"] == "LIW":
        if query_ids[0] == query_id:
            query_ids.pop(0)
            message_list = [f'These are the ranked documents, ordered from first to last, in epoch {epoch} in relation to the query mentioned:\n{all_docs}\n']

            if bot_info["ex_num"] > 1:
                for i in range(1, bot_info["ex_num"]):
                    all_docs = "\n\n".join(recent_data[recent_data.round_no == epoch - i].sort_values("position")["current_document"].values)
                    message_list.append(
                        f'Ranked documents, ordered from first to last, in epoch {epoch - i}:\n{all_docs}\n')
            message.append({'role': 'user', 'content': '\n\n'.join(message_list)})

        if query_ids:
            for qid in query_ids:
                message_list = []
                query_string = tops.iloc[0]['query']
                message_list.append(f'In the case of observing the query - {query_string}:\n')

                for i in range(bot_info["ex_num"]):
                    all_docs = "\n\n".join( data[(data.round_no == epoch - i) & (data['query_id'] == qid)].sort_values("position")["current_document"].values)
                    message_list.append(
                        f'Ranked documents, ordered from first to last, in epoch {epoch - i}:\n{all_docs}\n')
                message.append({'role': 'user', 'content': '\n'.join(message_list)})

    elif bot_info["method"] == "DYN":
        pass

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
    return message

if __name__ == '__main__':
    data = pd.read_csv("greg_data.csv")
    get_prompt("LIW_2101", data, 51, 195)
