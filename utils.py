from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def tokenize_data(dataset, tokenizer, max_len=256):
    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=max_len)
    return dataset.map(tokenize, batched=True)

def get_model():
    return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def get_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

def load_and_prepare_data():
    dataset = load_dataset("imdb", split='train[:10%]')
    tokenizer = get_tokenizer()
    tokenized = tokenize_data(dataset, tokenizer)
    small_dataset = tokenized.shuffle(seed=42).train_test_split(test_size=0.2)
    return small_dataset["train"], small_dataset["test"]