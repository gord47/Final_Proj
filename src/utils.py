import contractions
import emoji
import re, string
import nltk
import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset


nltk.download(['punkt_tab', 'wordnet', 'punkt', 'stopwords', 'averaged_perceptron_tagger'], quiet=True)

stop_words = set(stopwords.words('english'))

punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
digit_pattern = re.compile(r'\d+')
whitespace_pattern = re.compile(r'\s+')
non_word_pattern = re.compile(r'\W+')

class HateSpeechDataset(Dataset):

    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item['Cleaned_Content']
        label = item['Label']
        encoding = self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"],
            "labels": label,
            "text": text
        }
    

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = contractions.fix(text)
    text = digit_pattern.sub(' ', text)
    text = punctuation_pattern.sub(' ', text)
    text = non_word_pattern.sub(' ', text)
    text = whitespace_pattern.sub(' ', text).strip()
    text = emoji.demojize(text)
    return text


def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch]),
        'text': [x['text'] for x in batch]
    }