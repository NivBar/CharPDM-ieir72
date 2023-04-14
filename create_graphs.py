import matplotlib.pyplot as plt
import config


def create_markov_graphs(df):
    couples = set(df[["author_id", "query_id"]].itertuples(index=False, name=None))
    for a, q in couples:
        temp = df[(df.author_id == a) & (df.query_id == q)]
        min_, max_ = min(temp.round_number.astype(int)), max(temp.round_number.astype(int)) + 1

        # graphs for each author and query
        if config.display_graphs:
            x = list(range(min_, max_))
            plt.plot(x, temp.DELTA, label="DELTA")
            plt.plot(x, temp.POS, label="POS")
            plt.plot(x, temp.QREL, label="QREL")
            plt.plot(x, temp.KSREL, label="KSREL")
            plt.legend()
            plt.title(f"Author {a} Query {q}")
            plt.show()
