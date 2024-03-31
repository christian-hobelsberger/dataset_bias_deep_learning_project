import tensorflow as tf
from dataset_sampling import train_test_split
import numpy as np

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_save_models(datasets):
    for dataset_name, dataset in datasets.items():
        train_set, test_set = train_test_split(dataset['data'], label=dataset['label'])
        
        train_images = np.array([data['image'] for data in train_set])
        train_labels = np.array([1 if data['label'] == 'car' else 0 for data in train_set])
        test_images = np.array([data['image'] for data in test_set])
        test_labels = np.array([1 if data['label'] == 'car' else 0 for data in test_set])
        
        input_shape = train_images[0].shape
        
        model = create_cnn_model(input_shape)
        model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
        
        test_loss, test_accuracy = model.evaluate(test_images, test_labels)
        print(f"Test Accuracy for {dataset_name}:", test_accuracy)
        
        model.save(f"car_classifier_model_{dataset_name}.h5")

# Define your datasets here
datasets = {
    'dataset1': {'data': dataset1, 'label': 'car'},
    'dataset2': {'data': dataset2, 'label': 'car'},
    'dataset3': {'data': dataset3, 'label': 'car'},
    'dataset4': {'data': dataset4, 'label': 'car'},
    'dataset5': {'data': dataset5, 'label': 'car'},
    'dataset6': {'data': dataset6, 'label': 'car'}
}

train_and_save_models(datasets)