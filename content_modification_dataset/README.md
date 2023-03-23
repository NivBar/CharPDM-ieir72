# content_modification_dataset
A dataset described in the paper "Ranking-Incentivized Quality Preserving Content Modification"

# Data-set description
The collection contains 300 documents (rounds 1-4) created by human authors and competing bots.

Format: trectext.
DOCNO Format: ROUND-<round_number>-<query_id>-<author_id>

Docno's of round 0 are the initial documents given to the students.
Planted document have author_id of DUMMY_{1,2}.
Model's documents have author_id of BOT.

# Relevance Judgments (qrels):
All documents in the collection were judged for relevance. 
Annotators were presented with both the title and the description of each TREC topic
and were asked to classify a document as relevant if it satisfies the information need stated in the description.

A document judged relevant by less than three annotators was labeled as non-relevant (0).
Documents judged relevant by at least three, four or five annotators 
were labeled as marginally relevant (1), fairly relevant (2) and highly relevant (3), respectively.

# Quality judgements:
All documents in the collection where judged for quality by five annotators. 
Annotators were presented with the text of the document and were asked to classify the docuemnt as:
(1) Valid, (2) Keyword-stuffed, (3) Spam.

A document is deemed as keyword-stuffed if it contained excessive repetition of words 
which seemed unnatural or artificially introduced.

A document is considered as spam if its content could not possibly satisfy any information need.

If a document is not spam or keywordstuffed, it is considered as valid.

# Positions
The position in the ranked lists is found in the file "documents.positions".

File format:

qid 0 docno position

position 1 is the highest ranked document.

# Queries
We used 15 of ClueWeb09 queries which can be downloded here: http://trec.nist.gov/data/webmain.html. 

# Controlled experiment
Additional data for a controlled experiment we conducted with a slightly modified ranker.

# Citation
 If you use the ASRC dataset in your work, please cite it as:

  @inproceedings{Gorenetal20,
  author    = {Gregory Goren and
               Oren Kurland and
               Moshe Tennenholtz and
               Fiana Raiber},
  title     = {Ranking-Incentivized Quality Preserving Content Modification},
  booktitle = {Proceedings of the 43th International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, Virtual Event,
               China, July 25-30, 2020},
  pages     = {259-268},
  year      = {2020},
  crossref  = {DBLP:conf/sigir/2020},
  url       = {https://doi.org/10.1145/3397271.3401058},
  doi       = {10.1145/3397271.3401058},
}
