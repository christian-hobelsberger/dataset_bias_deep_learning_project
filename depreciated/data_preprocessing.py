import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (required by many pre-trained models)
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])

# Download and load the Caltech 101 dataset
train_dataset = torchvision.datasets.Caltech101(
    root='./data', 
    target_type='category',  # Using category as target type
    transform=transform, 
    download=True
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

# Initialize an iterator over the training dataset
train_loader_iter = iter(train_loader)

# Get a batch of data
images, labels = next(train_loader_iter)

# Print labels
print(labels)

import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Get the list of class names (subdirectories) from the root location of the dataset
class_names = sorted(os.listdir('./data/caltech101/Annotations'))

# Let's visualize some images and their corresponding labels
num_images_to_display = 5

# Iterate over the dataset and display some images
for i in range(num_images_to_display):
    # Get a random image and its label
    image = images[i]
    label = labels[i]
    
    # Convert tensor to numpy array and transpose dimensions
    image = image.permute(1, 2, 0).numpy()
    
    # Display the image
    plt.imshow(image)
    plt.title(f'Label: {class_names[label]}')
    plt.axis('off')
    plt.show()

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