import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import current_prompt as cp

file_path = f'/lv_local/home/niv.b/content_modification_code-master/Results/RankedLists/LambdaMART{cp}'

columns = ['query_id', 'Q0', 'docno', 'rank', 'score', 'method']

with open(file_path, 'r') as file:
    lines = file.readlines()

data = []

for line in lines:
    values = line.strip().split()
    query_id, _, document_id, rank, score, method = values
    data.append([query_id, 'Q0', document_id, int(rank), float(score), method])

df = pd.DataFrame(data, columns=columns).drop(["Q0", "method"], axis=1)

#### Features preprocess ####
folder_path = f'/lv_local/home/niv.b/content_modification_code-master/Results/Features/{cp}'

# Iterate over files in the folder
for filename in tqdm(os.listdir(folder_path)):
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
            docno, sum_, min_, max_, mean_, var_ = values
            df.loc[df['docno'] == docno, feat] = mean_

df.to_csv(f"feature_data_{cp}.csv", index=False)

#### visualization ####

cols = ['rank', 'score', 'docCoverQueryNum',
        'docBoolean.OR', 'docEnt', 'docVSM', 'docFracStops', 'docTF', 'docBM25',
        'docTFIDF', 'docCoverQueryRatio', 'docIDF', 'docTFNorm',
        'docBoolean.AND', 'docLMIR.JM', 'docLMIR.DIR', 'docStopCover', 'docLen',
        'docLMIR.ABS']

thresh = 0.2
label = "score"

correlations = df[cols].corr()[label]

correlations = pd.DataFrame(correlations)  # Convert to DataFrame
correlations = correlations[(np.abs(correlations[label]) >= thresh) & (correlations[label] != 1)].reindex(
    np.abs(correlations[(np.abs(correlations[label]) >= thresh) & (correlations[label] != 1)][label]).sort_values(
        ascending=False).index)
# correlations = correlations[(np.abs(correlations[label]) >= thresh) & (correlations[label] != 1)]

g = sns.heatmap(correlations, annot=True, cmap="RdYlGn", vmin=-1, vmax=1)
g.set_title("Rank Correlations")

plt.tight_layout()
plt.show()
