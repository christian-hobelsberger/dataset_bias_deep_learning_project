import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess the images and labels
data_dir = 'data/msrcorid/'
images_list = []
labels_list = []
processed_files = set()

# Counter for images added per folder
images_added_per_folder = {}

def process_folder(folder_path, label):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isdir(file_path):
            process_folder(file_path, label)  # Recursively process subdirectories
        else:
            # Check if file has already been processed
            if file_path not in processed_files:
                processed_files.add(file_path)  # Add file to processed set
                try:
                    img = image.load_img(file_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    images_list.append(img_array)
                    labels_list.append(label)
                    images_added_per_folder[os.path.dirname(file_path)] = images_added_per_folder.get(os.path.dirname(file_path), 0) + 1  # Increment counter
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

# Loop through top-level directories
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        if folder_name == 'cars':
            label = 1
        else:
            label = 0

        print(f"Processing directory: {folder_path}, Label: {label}")

        # Process the folder
        process_folder(folder_path, label)

# Convert lists to arrays
images = np.array(images_list)
labels = np.array(labels_list)

# Count number of cars and non-cars
num_cars = np.sum(labels == 1)
num_non_cars = np.sum(labels == 0)

print("Number of labels with value 1 (cars):", num_cars)
print("Number of labels with value 0 (non-cars):", num_non_cars)

# Print images added per folder
print("\nImages added per folder:")
for folder_name, count in images_added_per_folder.items():
    print(f"{folder_name}: {count} images added")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Adding dropout for regularization
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=binary_crossentropy, metrics=[Accuracy()])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Evaluate the model
loss, acc = model.evaluate(X_val, y_val)
print('Validation Accuracy:', acc)
print('Validation Loss:', loss)

# Save the trained model
model.save("models\msrcorid_model_cnn_adj.keras")

# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()

# Create folder if it doesn't exist
plots_folder = 'plots'
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Save plots
accuracy_plot_path = os.path.join(plots_folder, 'accuracy_plot_adj_msrcorid.png')
loss_plot_path = os.path.join(plots_folder, 'loss_plot_adj_msrcorid.png')
plt.savefig(accuracy_plot_path)
plt.savefig(loss_plot_path)