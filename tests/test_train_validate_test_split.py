from unittest import TestCase
import numpy as np
from notebooks.utils import train_validate_test_split


class TestTrainValidateTestSplit(TestCase):
    def test_stratify(self):
        row_count = 10000
        eps = 0.005
        x = (np.random.randint(0, 20, row_count) == 13).astype(int)
        y = x.copy()
        y_class_1, y_class_2 = np.unique(y, return_counts=True)[1]
        x_train, x_validate, x_test, y_train, y_validate, y_test = train_validate_test_split(x, y, train_size=.33,
                                                                                             test_validate_ratio=1,
                                                                                             random_state=42,
                                                                                             stratify=y)
        y_train_class_1, y_train_class_2 = np.unique(y_train, return_counts=True)[1]
        y_validate_class_1, y_validate_class_2 = np.unique(y_train, return_counts=True)[1]
        y_test_class_1, y_test_class_2 = np.unique(y_train, return_counts=True)[1]

        self.assertTrue(np.abs(1 - (y_class_1 / y_class_2) / (y_train_class_1 / y_train_class_2)) < eps)
        self.assertTrue(np.abs(1 - (y_class_1 / y_class_2) / (y_validate_class_1 / y_validate_class_2)) < eps)
        self.assertTrue(np.abs(1 - (y_class_1 / y_class_2) / (y_test_class_1 / y_test_class_2)) < eps)

        self.assertTrue(np.all(x_train == y_train))
        self.assertTrue(np.all(x_validate == y_validate))
        self.assertTrue(np.all(x_test == x_test))

    def test_len(self):
        row_count = 10000
        x = np.random.random((row_count, 1))
        y = np.random.randint(0, 2, row_count)
        x_train, x_validate, x_test, y_train, y_validate, y_test = train_validate_test_split(x, y, train_size=.33,
                                                                                             test_validate_ratio=1,
                                                                                             random_state=42)
        self.assertTrue(x.shape[0] == np.sum([x_train.shape[0], x_validate.shape[0], x_test.shape[0]]))
        self.assertTrue(x_train.shape[0] == y_train.shape[0])
        self.assertTrue(x_validate.shape[0] == y_validate.shape[0])
        self.assertTrue(x_test.shape[0] == y_test.shape[0])

    def test_validate_ratio(self):
        row_count = 10000
        eps = 0.01
        validate_ratios = [0.5, 1, 1.5, 2]
        for validate_ratio in validate_ratios:
            x = np.random.random((row_count, 1))
            x_train, x_validate, x_test = train_validate_test_split(x, train_size=.33,
                                                                    test_validate_ratio=validate_ratio,
                                                                    random_state=42)
            self.assertTrue(x_validate.shape[0] / x_test.shape[0] - validate_ratio < eps)

    def test_train_size(self):
        row_count = 10000
        eps = 0.01
        train_sizes = np.arange(0.2, 0.91, 0.1)
        for train_size in train_sizes:
            x = np.random.random((row_count, 1))
            x_train, x_validate, x_test = train_validate_test_split(x, train_size=train_size,
                                                                    test_validate_ratio=1,
                                                                    random_state=42)
            self.assertTrue(x_train.shape[0] / row_count - train_size < eps)
