from pathlib import Path
import pandas as pd
import gensim

from nltk.tokenize import word_tokenize

d2v_model_path = Path("/Users/David/Documents/CS/MPhil_Project/embeddings/enwiki_dbow/doc2vec.bin")
tractatus_path = Path("tractatus.csv")

def load_tractatus(tractatus_path=tractatus_path):
    return pd.read_csv(tractatus_path.as_posix())


def create_d2v_embeddings(tractatus_df, model_path=d2v_model_path):
    d2v = gensim.models.Doc2Vec.load(model_path.as_posix())


    embeddings = [d2v.infer_vector(word_tokenize(row["text"])) for _, row in tractatus_df]

    print(embeddings[42])


if __name__ == "__main__":
    df = load_tractatus()
    create_d2v_embeddings(df)
