from transformers import Wav2Vec2CTCTokenizer
from huggingface_hub import notebook_login


repo_name = "wav2vec2-base-timit-demo-colab"



tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")




print(tokenizer)