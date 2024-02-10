import torch


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, balanced=True):
        self.balanced = balanced
        self.labels = labels
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer(text,
                              padding='max_length',
                              max_length=self.max_length,
                              truncation=True,
                              return_tensors='pt')
        label = self.labels[idx]
        return text['input_ids'], text['attention_mask'], label


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length, balanced=True):
        self.balanced = balanced
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
