import torch


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
