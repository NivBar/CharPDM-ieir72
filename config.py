import json

import openai

data_exist = {"comp": True, "improvements": False, "tops": False}
display_graphs = False

openai.api_key = "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"

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
model = "gpt-3.5-turbo"
temperature = 1.0
top_p = 0.9
max_tokens = 700
frequency_penalty = 1.0
presence_penalty = 2.0

topic_codex = json.load(open("topic_queries_doc.json", "r"))
x = 1



