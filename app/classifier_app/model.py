import json
import os
from dataset import InferenceDataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gdown
import torch
from torch.utils.data import DataLoader
from transformers import logging

logging.set_verbosity_error()

current_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(current_dir, 'config.json')
with open(config_path) as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):
        self.device = torch.device(config['DEVICE'] if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
        self.max_length = config['MAX_LENGTH']
        self.batch_size = config['BATCH_SIZE']
        self.classifier = AutoModelForSequenceClassification.from_pretrained(config['MODEL_NAME'], num_labels=3)
        self.classifier = self.classifier.to(self.device)
        if not os.path.exists('models'):
            os.mkdir('models')
        full_path = os.path.join('models', config['PRE_TRAINED_MODEL'])
        if not os.path.exists(full_path):
            if not config['GDRIVE_URL']:
                raise ValueError('No model checkpoint.')
            gdown.download(config['GDRIVE_URL'], full_path, fuzzy=True)
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
        confidence, predicted_classes = torch.max(probabilities, dim=1)
        confidence = confidence.tolist()
        predicted_classes = predicted_classes.tolist()
        return {
            'sentiments': [config["CLASS_NAMES"][predicted_class] for predicted_class in predicted_classes],
            'confidences': confidence,
            'probabilities': dict(zip(config["CLASS_NAMES"], probabilities.T.tolist())),
        }


model = Model()
