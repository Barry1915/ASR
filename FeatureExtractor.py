import IPython.display as ipd
import numpy as np
import random
import pandas as pd
from datasets import load_dataset
import re

from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)



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


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch



if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 10000)

    timit = load_dataset("timit_asr", data_dir=r'C:\Users\yst\PycharmProjects\Transformer\data')

    timit = timit.remove_columns(
        ["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

    timit = timit.map(remove_special_characters)

    rand_int = random.randint(0, len(timit["train"]))

    ipd.Audio(data=np.asarray(timit["train"][rand_int]["audio"]["array"]), autoplay=True, rate=16000)

    print("Target text:", timit["train"][rand_int]["text"])
    print("Input array shape:", np.asarray(timit["train"][rand_int]["audio"]["array"]).shape)
    print("Sampling rate:", timit["train"][rand_int]["audio"]["sampling_rate"])
    timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4)

    print(timit)
