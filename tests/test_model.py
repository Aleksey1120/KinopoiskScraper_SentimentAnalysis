import sys

sys.path.append('apps/review_classifier')

import numpy as np
from unittest import TestCase

from apps.review_classifier.model import Model


class TestModel(TestCase):
    def test_sentiment(self):
        eps = 0.001
        model = Model()

        reviews = ['Очень хороший фильм. Рекомендую.',
                   'Захватывающее начало, но скомканный финал.',
                   'Скучный сюжет, шаблонные персонажи, предсказуемый финал. Не тратьте время.']
        expected_sentiment = ['positive', 'neutral', 'negative']

        predicts = model.predict(reviews)
        self.assertTrue(len(predicts) == len(reviews))
        for predict, expected in zip(predicts, expected_sentiment):
            self.assertTrue('sentiment' in predict)
            self.assertTrue('confidence' in predict)
            self.assertTrue('probability' in predict)

            self.assertTrue(predict['sentiment'] == expected)

            self.assertTrue(np.abs(sum(predict['probability'].values()) - 1.0) < eps)

            max_class = max(predict['probability'].items(), key=lambda x: x[1])

            self.assertTrue(max_class[0] == predict['sentiment'])
            self.assertTrue(max_class[1] == predict['confidence'])
