import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging
import gdown
import torch
from torch.utils.data import DataLoader

from apps.review_classifier.dataset import InferenceDataset
from apps.review_classifier import config

logging.set_verbosity_error()


class Model:
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.max_length = config.MAX_LENGTH
        self.batch_size = config.BATCH_SIZE
        self.classifier = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=3)
        self.classifier = self.classifier.to(self.device)
        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)
        full_path = os.path.join(config.MODEL_DIR, config.PRE_TRAINED_MODEL)
        if not os.path.exists(full_path):
            if not config.GDRIVE_URL:
                raise ValueError('No model checkpoint.')
            gdown.download(config.GDRIVE_URL, full_path, fuzzy=True)
        self.classifier.load_state_dict(torch.load(full_path, map_location=self.device))
        self.classifier.eval()

    def predict(self, texts: list[str]):
        dataset = InferenceDataset(texts, self.tokenizer, self.max_length)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        probabilities = []
        with torch.no_grad():
            for batch in loader:
                input_id, mask = batch
                mask = mask.to(self.device)
                input_id = input_id.squeeze(1).to(self.device)
                output = self.classifier(input_id, attention_mask=mask).logits
                probabilities.append(torch.nn.functional.softmax(output, dim=1))
        probabilities = torch.cat(probabilities).cpu()
        confidences, predicted_classes = torch.max(probabilities, dim=1)
        confidences = confidences.tolist()
        predicted_classes = predicted_classes.tolist()
        probabilities = probabilities.tolist()

        results = []
        for predicted_class, confidence, probas in zip(predicted_classes, confidences, probabilities):
            results.append({
                'sentiment': config.CLASS_NAMES[predicted_class],
                'confidence': confidence,
                'probability': dict(zip(config.CLASS_NAMES, probas))
            })

        return results
