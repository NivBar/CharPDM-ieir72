import json

from pymongo import MongoClient

inital_data = json.load(open("topic_queries_doc.json", "r", encoding="utf8"))
client = MongoClient('asr2.iem.technion.ac.il', 27017)
db = client.asr16

init = db["initials"]
for k, v in inital_data.items():
    row = {"_id": k}
    row.update(v)
    init.insert_one(row)
