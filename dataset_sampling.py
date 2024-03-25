import random

def random_sample_dataset(dataset, n, label=None, label_prop=None):
    if label and label_prop:
        label_count = sum(1 for data in dataset if data['label'] == label)
        min_label_count = int(n * label_prop)
        if label_count < min_label_count:
            raise ValueError(f"The dataset does not have enough samples with label '{label}' to meet the minimum proportion requirement.")
        filtered_dataset = [data for data in dataset if data['label'] == label]
        remaining_samples = n - min_label_count
        sampled_dataset = random.sample(filtered_dataset, remaining_samples)
        sampled_dataset.extend(random.sample(dataset, min_label_count))
    else:
        sampled_dataset = random.sample(dataset, n)
    return sampled_dataset