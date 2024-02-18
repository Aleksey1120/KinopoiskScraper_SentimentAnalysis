from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def get_model_and_tokenizer(model_name_or_path, checkpoint_path, cache_dir=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                               num_labels=3,
                                                               cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              cache_dir=cache_dir)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    return model, tokenizer
