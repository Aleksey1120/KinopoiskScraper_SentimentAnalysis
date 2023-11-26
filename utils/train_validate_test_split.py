from sklearn.model_selection import train_test_split


def train_validate_test_split(*arrays, train_size: float = 0.8,
                              test_validate_ratio: float = 1.0,
                              random_state=None,
                              stratify: int = None):
    arrays_count = len(arrays)
    split_arrays = train_test_split(*arrays,
                                    train_size=train_size,
                                    random_state=random_state,
                                    stratify=stratify if stratify is not None else None)
    train = split_arrays[::arrays_count]
    tmp = split_arrays[1::arrays_count]
    validate_size = test_validate_ratio / (1 + test_validate_ratio)
    if stratify is not None:
        _, tmp_stratify = train_test_split(stratify, train_size=train_size, random_state=random_state,
                                           stratify=stratify)
    split_arrays = train_test_split(*tmp,
                                    random_state=random_state,
                                    train_size=validate_size,
                                    stratify=tmp_stratify if stratify is not None else None)
    validate = split_arrays[::arrays_count]
    test = split_arrays[1::arrays_count]
    for i in range(arrays_count):
        yield train[i]
        yield validate[i]
        yield test[i]
