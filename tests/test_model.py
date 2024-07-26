import sys

sys.path.append('app/classifier_app')

import pandas as pd
import numpy as np
from unittest import TestCase

from app.classifier_app.model import Model


class TestModel(TestCase):
    def test_sentiment(self):
        eps = 0.001
        model = Model()

        reviews = ['Очень хороший фильм. Рекомендую.',
                   'Захватывающее начало, но скомканный финал.',
                   'Скучный сюжет, шаблонные персонажи, предсказуемый финал. Не тратьте время.']

        predict = model.predict(reviews)
        self.assertTrue('sentiments' in predict)
        self.assertTrue('confidences' in predict)
        self.assertTrue('probabilities' in predict)

        self.assertTrue(predict['sentiments'] == ['positive', 'neutral', 'negative'])
        self.assertTrue(len(predict['confidences']) == 3)
        self.assertTrue(all([pred > 0.33 for pred in predict['confidences']]))

        probabilities = pd.DataFrame(predict['probabilities'])
        self.assertTrue(probabilities.shape == (3, 3))
        self.assertTrue(set(probabilities.columns) == {'negative', 'neutral', 'positive'})
        self.assertTrue(np.all(np.abs(np.sum(probabilities, axis=1) - 1.0) < eps))

        self.assertTrue(np.all(np.abs(probabilities.max(axis=1) - predict['confidences']) < eps))
