from pathlib import Path
import pandas as pd
import logging
import csv

from nltk.tokenize import sent_tokenize

import torch
from pytorch_transformers import BertTokenizer, BertModel, BertConfig

logging.basicConfig(level=logging.DEBUG)


class tractatus_embeder(object):
    def __init__(self, path_to_csv="tractatus_with_splits.csv", pretrained_name="bert-base-cased", n_layers=4, start_split="train"):
        """

        :type n_layers: int
        """
        self.n_layers = n_layers
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        our_config = BertConfig(vocab_size_or_config_json_file=28996, output_hidden_states=True)
        self.model = BertModel.from_pretrained(pretrained_name, config=our_config)
        self.model.eval()

        self.all_df = pd.read_csv(path_to_csv)

        self.test_df = self.all_df[self.all_df.split == "test"]
        self.train_df = self.all_df[self.all_df.split == "train"]
        self.val_df = self.all_df[self.all_df.split == "validation"]

        self._lookup_dict = {"train": self.train_df,
                             "val": self.val_df,
                             "test": self.test_df}

        self.set_split(start_split)

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df = self._lookup_dict[split]

    @staticmethod
    def prepare_statement(text):
        sentences = sent_tokenize(text)
        return ["[CLS] " + sent + " [SEP]" for sent in sentences]

    def embed_statement(self, statement):
        """

        :rtype: pytorch tensor object
        """
        logging.debug(f"Statement: {statement}")

        token_embeddings = []  # already introduced here because we accumulate them over sentences
        for sent in self.prepare_statement(statement):
            # logging.debug(f"Sentence: {sent}")

            encoded_statement = self.tokenizer.encode(statement)
            # logging.debug(f"Encoded statement sentence: {encoded_statement}")

            input_ids = torch.tensor([encoded_statement])
            # logging.debug(f"input_ids: {input_ids}")

            segments_ids = torch.tensor([1] * len(input_ids))
            # logging.debug(f"segment_ids: {segments_ids}")

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=segments_ids)

                # logging.debug(f"length of outputs: {len(outputs)}")

                last_hidden_state = outputs[0]  # last layer
                encoded_layers = (last_hidden_state,) + outputs[2]  # add the last layer to all the others

                assert tuple(encoded_layers[0].shape) == (1, len(encoded_statement), self.model.config.hidden_size)

                # logging.debug(f"Number of layers: {len(encoded_layers)}")
                # logging.debug(f"Number of batches: {len(encoded_layers[0])}")
                # logging.debug(f"Number of tokens:, {len(encoded_layers[0][0])}")
                # logging.debug(f"Number of hidden units:, {len(encoded_layers[0][0][0])}")

                batch_index = 0
                for token_index in range(len(encoded_statement)):
                    hidden_layers = []

                    # For each of the layers
                    for layer_index in range(len(encoded_layers)):
                        # Lookup the vector for `token_index` in `layer_index`
                        vector = encoded_layers[layer_index][batch_index][token_index]

                        hidden_layers.append(vector)

                    token_embeddings.append(hidden_layers)

            summed_last_n_layers = [torch.sum(torch.stack(layer)[-self.n_layers:], 0) for layer in
                                    token_embeddings]
            # [number_of_tokens, 768]

            statement_embedding = torch.mean(torch.stack(summed_last_n_layers, 1), 1)
            # logging.debug(len(summed_last_n_layers))
            # logging.debug(f"statement embedding: {statement_embedding}")
            # logging.debug(f"Shape of embedding: {statement_embedding.shape}")

            return statement_embedding

    def create_next_embedding(self, all_splits=False):
        if all_splits:
            df = self.all_df
        else:
            df = self._target_df

        for _, row in df.iterrows():
            embedding = self.embed_statement(row["text"])

            yield embedding, row

    def write_embeddings_to_file(self, main_path="embeddings.tsv", meta_path="embeddings_meta.tsv"):
        with open(main_path, "w", newline="") as out_file, open(meta_path, "w", newline="") as meta_file:
            out_writer = csv.writer(out_file, delimiter="\t")
            meta_writer = csv.writer(meta_file, delimiter="\t")

            meta_writer.writerow(["number", "depth"])
            for embedding, row in self.create_next_embedding():
                out_writer.writerow(embedding.tolist())
                meta_writer.writerow([str(row["number"]), row["depth"]])

if __name__ == "__main__":
    te = tractatus_embeder()
    te.write_embeddings_to_file()
