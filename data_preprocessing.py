import pandas as pd

# List of file paths for the CSV datasets
file_paths = [
    "data/dataset1.csv",
    "data/dataset2.csv",
    "data/dataset3.csv",
    "data/dataset4.csv",
    "data/dataset5.csv",
    "data/dataset6.csv"
]

# Read in each CSV file separately and save as separate variables
dataset1 = pd.read_csv(file_paths[0])
dataset2 = pd.read_csv(file_paths[1])
dataset3 = pd.read_csv(file_paths[2])
dataset4 = pd.read_csv(file_paths[3])
dataset5 = pd.read_csv(file_paths[4])
dataset6 = pd.read_csv(file_paths[5])

def add_label_binary_column(dataset):
    dataset['label_binary'] = dataset['label'].str.contains('car')
    return dataset

# Example usage:
dataset1 = add_label_binary_column(dataset1)
dataset2 = add_label_binary_column(dataset2)
dataset3 = add_label_binary_column(dataset3)
dataset4 = add_label_binary_column(dataset4)
dataset5 = add_label_binary_column(dataset5)
dataset6 = add_label_binary_column(dataset6)

# Save the preprocessed datasets in the data/ folder with the extension _preprocessed
dataset1.to_csv('data/dataset1_preprocessed.csv', index=False)
dataset2.to_csv('data/dataset2_preprocessed.csv', index=False)
dataset3.to_csv('data/dataset3_preprocessed.csv', index=False)
dataset4.to_csv('data/dataset4_preprocessed.csv', index=False)
dataset5.to_csv('data/dataset5_preprocessed.csv', index=False)
dataset6.to_csv('data/dataset6_preprocessed.csv', index=False)