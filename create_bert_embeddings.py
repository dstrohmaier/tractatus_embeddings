from pathlib import Path
import pandas as pd
import logging

from nltk.tokenize import sent_tokenize

import torch
from pytorch_transformers import BertTokenizer, BertModel
from torch import Tensor

logging.basicConfig(level=logging.INFO)


tractatus_path = Path("tractatus.csv")

def load_tractatus(tractatus_path=tractatus_path):
    return pd.read_csv(tractatus_path.as_posix())



class tractatus_embeder(object):
    def __init__(self, path_to_csv="tractatus.csv", pretrained_name="bert-base-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        self.model = BertModel.from_pretrained(pretrained_name)

        self.df = pd.read_csv(path_to_csv)

    @staticmethod
    def insert_special_tokens(text):
        sentences = sent_tokenize(text)
        return "[CLS] " + " [SEP] ".join(sentences) + " [SEP]"

    def embed_statement(self, statement):
        input_ids = torch.tensor([self.tokenizer.encode(statement)])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
            return last_hidden_states

    def create_next_embedding(self):
        for i, row in self.df.iterrows():
            embedding = self.embed_statement(row["statement"])
            print(embedding)
            yield embedding
            if i > 3: # for testing purposes
                break




if __name__ == "__main__":
    te = tractatus_embeder()
    while True:
        te.create_next_embedding()
