import random
import pandas as pd
from datasets import load_dataset, load_metric
import re
import json


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    # display(HTML(df.to_html()))
    print(df)


def remove_special_characters(batch):
    # batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}





if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 10000)

    timit = load_dataset("timit_asr", data_dir=r'C:\Users\yst\PycharmProjects\Transformer\data')

    timit = timit.remove_columns(
        ["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])


    timit = timit.map(remove_special_characters)


    vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                       remove_columns=timit.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    print(vocab_dict)

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    print(vocab_dict)

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(vocab_dict)

    print(len(vocab_dict))

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


