import json

import pandas as pd
import numpy as np
import requests
from unittest import TestCase


class TestClassifierAPI(TestCase):
    def test_api(self):
        eps = 0.001

        reviews = ['Очень хороший фильм. Рекомендую.',
                   'Захватывающее начало, но скомканный финал.',
                   'Скучный сюжет, шаблонные персонажи, предсказуемый финал. Не тратьте время.']

        response = requests.post('http://localhost:8080/predict', json.dumps({'texts': reviews})).json()
        self.assertTrue('sentiments' in response)
        self.assertTrue('confidences' in response)
        self.assertTrue('probabilities' in response)

        self.assertTrue(response['sentiments'] == ['positive', 'neutral', 'negative'])
        self.assertTrue(len(response['confidences']) == 3)
        self.assertTrue(all([pred > 0.33 for pred in response['confidences']]))

        probabilities = pd.DataFrame(response['probabilities'])
        self.assertTrue(probabilities.shape == (3, 3))
        self.assertTrue(set(probabilities.columns) == {'negative', 'neutral', 'positive'})
        self.assertTrue(np.all(np.abs(np.sum(probabilities, axis=1) - 1.0) < eps))

        self.assertTrue(np.all(np.abs(probabilities.max(axis=1) - response['confidences']) < eps))
