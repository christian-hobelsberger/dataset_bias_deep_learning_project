import random

def train_test_split(dataset, label, train_positive_size=500, train_negative_size=2000, test_positive_size=50, test_negative_size=1000):
    positive_samples = [data for data in dataset if data['label'] == label]
    negative_samples = [data for data in dataset if data['label'] != label]

    if len(positive_samples) < train_positive_size or len(negative_samples) < train_negative_size:
        raise ValueError("The dataset does not have enough positive or negative samples to meet the minimum size requirement for the train set.")

    if len(positive_samples) < test_positive_size or len(negative_samples) < test_negative_size:
        raise ValueError("The dataset does not have enough positive or negative samples to meet the minimum size requirement for the test set.")

    train_positive_samples = random.sample(positive_samples, train_positive_size)
    train_negative_samples = random.sample(negative_samples, train_negative_size)
    test_positive_samples = random.sample(list(set(positive_samples) - set(train_positive_samples)), test_positive_size)
    test_negative_samples = random.sample(list(set(negative_samples) - set(train_negative_samples)), test_negative_size)

    train_set = train_positive_samples + train_negative_samples
    test_set = test_positive_samples + test_negative_samples

    return train_set, test_set
