import os.path

import torch
from diskcache import Cache


class TokensCache:

    def __init__(self, tokenizer, max_length):
        self.cache = Cache(os.path.join('cache_dir', tokenizer.name_or_path, str(max_length)))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_tokens(self, text):
        if text in self.cache:
            return self.cache[text]
        tokens = self.tokenizer(text,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_tensors='pt')
        self.cache[text] = tokens['input_ids'], tokens['attention_mask']
        return tokens['input_ids'], tokens['attention_mask']

    def close(self):
        self.cache.close()


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, use_cache=False):
        self.labels = labels
        self.texts = texts

        if use_cache:
            self.tokens_cache = TokensCache(tokenizer, max_length)
        else:
            self.tokens_cache = None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.tokens_cache is not None:
            input_ids, attention_mask = self.tokens_cache.get_tokens(text)
        else:
            tokens = self.tokenizer(text,
                                    padding='max_length',
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_tensors='pt')
            input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        label = self.labels[idx]
        return input_ids, attention_mask, label

    def __del__(self):
        if self.tokens_cache is not None:
            self.tokens_cache.close()


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer(text,
                              padding='max_length',
                              max_length=self.max_length,
                              truncation=True,
                              return_tensors='pt')
        return text['input_ids'], text['attention_mask']


class Fetcher:

    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.data = iter(self.loader)

    def load(self):
        try:
            return next(self.data)
        except StopIteration:
            self.reset()
            return next(self.data)
