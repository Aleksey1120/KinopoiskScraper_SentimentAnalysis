import json

import numpy as np
import requests
from unittest import TestCase


class TestClassifierAPI(TestCase):
    def test_api(self):
        eps = 0.001

        review = 'Очень хороший фильм. Рекомендую.'

        response = requests.post('http://localhost:8080/predict', json.dumps({'text': review})).json()

        self.assertTrue('sentiment' in response)
        self.assertTrue('confidence' in response)
        self.assertTrue('probability' in response)

        self.assertTrue(response['sentiment'] == 'positive')

        self.assertTrue(np.abs(sum(response['probability'].values()) - 1.0) < eps)

        max_class = max(response['probability'].items(), key=lambda x: x[1])

        self.assertTrue(max_class[0] == response['sentiment'])
        self.assertTrue(max_class[1] == response['confidence'])
