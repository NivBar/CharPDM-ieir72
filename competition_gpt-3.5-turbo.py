import openai
import config
import tiktoken

encoder = tiktoken.encoding_for_model(config.model)


def get_messages(idx: int, epoch=None, bot_type="all", markov=False):
    """
    craft prompt messages list for the bot according to its type
    :param idx: topic index
    :param epoch: current epoch number
    :param bot_type: "all" gives the bot documents and rankings along all epochs,
                     "tops" gives the bot only the top document along all epochs,
                     "self" gives the bot only the ranking it is in, in every epoch along all epochs
    :param markov: looking only on the last round
    :return: list of messages comprising the prompt
    """
    assert type(idx) == int
    assert epoch is None or type(epoch) == int
    assert bot_type in ["all", "tops", "self"]
    assert type(markov) == bool

    # topic_info = config.topic_codex[idx]

    # csv option:
    topic_info = {}
    rel = config.comp_data[config.comp_data.query_id == idx].head(1)
    topic_info['queries'] = [rel["query"][0]]
    topic_info['doc'] = rel['TEXT'][0]

    messages = [
        {"role": "system", "content": f"You are a contestant in an information retrieval SEO competition."},
        {"role": "system",
         "content": fr"The competition involves {len(topic_info['queries'])} queries: " + ",".join(
             topic_info['queries'])},
        {"role": "system",
         "content": "The goal is to have your document be ranked 1 (first) and win in the ranking done by a black box ranker."},
        {"role": "system", "content": "You can only generate texts of 150 words maximum."},
        {"role": "system", "content": fr"All contestants got an initial reference text: {topic_info['doc']}."},
        {"role": "user",
         "content": "Generate a single text that addresses the information need for all queries."},
    ]
    if epoch is not None:
        min_ = epoch if markov else 1
        for i in range(min_, epoch + 1):
            epoch_data = config.comp_data[
                (config.comp_data.query_id == idx) & (config.comp_data.round_number == i)].sort_values('POS',
                                                                                                       ascending=True)
            bot_data = epoch_data[epoch_data.author_id == config.names[bot_type]].reset_index()
            messages.append({"role": "assistant", "content": f"{bot_data['TEXT'][0]}"})
            messages.append({"role": "system", "content": f"You were ranked {bot_data['POS'][0]} in this epoch"})

            if bot_type == "all":
                txt_rnk_lst = []
                for _, row in epoch_data.iterrows():
                    if row['author_id'] == config.names[bot_type]: continue
                    txt_rnk_lst.append(f"ranked {row['POS']}: {row['TEXT']}\n\n")
                txt_rnk = "".join(txt_rnk_lst)

                messages.append(
                    {"role": "system",
                     "content": f"The documents of your opponents in this epoch are as follows:\n {txt_rnk}"})
            elif bot_type == "tops":
                top_data = epoch_data[epoch_data.POS == 1].reset_index()
                messages.append(
                    {"role": "system", "content": f"The document ranked 1 in this epoch is: {top_data['TEXT'][0]}"})
            elif bot_type == "self":
                pass
            messages.append(
                {"role": "user",
                 "content": "Generate a single text that addresses the information need for all queries."})
    print(messages)
    return messages


def get_comp_text(messages):
    max_tokens = config.max_tokens
    response = False
    prompt_tokens = len(encoder.encode("".join([line['content'] for line in messages]))) + 200
    while prompt_tokens + max_tokens > 4096:
        max_tokens -= 50
    print("max tokens for response:", max_tokens)

    while not response:
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=max_tokens,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )
            # print("success")
            word_no = len(response['choices'][0]['message']['content'].split())
            if word_no > 150:
                max_tokens -= 50
                response = False
                print(f"word no: {word_no}, max tokens: {max_tokens}.")
                continue
            break
        except Exception as e:
            print(e)
            continue
    return response


if __name__ == '__main__':
    messages = get_messages(193, epoch=4, bot_type="all", markov=False)
    res = get_comp_text(messages)['choices'][0]['message']['content']
    x = 1
