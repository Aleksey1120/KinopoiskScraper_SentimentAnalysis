from transformers import AutoModelForSequenceClassification, AutoTokenizer


# TODO: add checkpoint loading
def get_model_and_tokenizer(model_name_or_path, cache_dir=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                               num_labels=3,
                                                               cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              cache_dir=cache_dir)
    return model, tokenizer
