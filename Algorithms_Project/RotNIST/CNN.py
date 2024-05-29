import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import legacy as keras_legacy_optimizers
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    """
    Load dataset from a compressed .npz file.
    
    Args:
        filepath (str): Path to the .npz file containing the dataset.
    
    Returns:
        tuple: Tuple containing training, development, test, and challenge datasets.
    """
    with np.load(filepath) as data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_dev = data['X_dev']
        y_dev = data['y_dev']
        X_test = data['X_test']
        y_test = data['y_test']
        X_challenge = data['X_challenge']
        y_challenge = data['y_challenge']
    return X_train, y_train, X_dev, y_dev, X_test, y_test, X_challenge, y_challenge

def create_cnn_model(input_shape, num_classes):
    """
    Define and compile the Convolutional Neural Network (CNN) model.
    
    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
    
    Returns:
        Sequential: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def display_sample_images(images, labels, predictions, sample_size=5, correct=True):
    """
    Display sample images along with their true and predicted labels.
    
    Args:
        images (ndarray): Array of image data.
        labels (ndarray): Array of true labels.
        predictions (ndarray): Array of predicted labels.
        sample_size (int): Number of images to display.
        correct (bool): Whether to display correctly classified images or not.
    """
    plt.figure(figsize=(10, 2))
    count = 0
    for i in range(len(images)):
        if (correct and labels[i] == predictions[i]) or (not correct and labels[i] != predictions[i]):
            plt.subplot(1, sample_size, count + 1)
            plt.imshow(images[i].reshape(28, 28), cmap='gray')
            plt.title(f"Label: {labels[i]}\nPred: {predictions[i]}")
            plt.axis('off')
            count += 1
            if count == sample_size:
                break
    plt.show()

def main(args):
    """
    Main function to load data, train and evaluate the CNN model, and display results.
    
    Args:
        args (Namespace): Parsed command line arguments.
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test, X_challenge, y_challenge = load_data(args.data_path)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = create_cnn_model(input_shape, num_classes)
    model.compile(optimizer=keras_legacy_optimizers.Adam(learning_rate=args.learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    challenge_dataset = tf.data.Dataset.from_tensor_slices((X_challenge, y_challenge)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    challenge_loss, challenge_accuracy = model.evaluate(challenge_dataset)
    print(f"Challenge loss: {challenge_loss}")
    print(f"Challenge accuracy: {challenge_accuracy}")

    y_pred = model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Correctly classified images:")
    display_sample_images(X_test, y_test, y_pred_classes, correct=True)

    print("Incorrectly classified images:")
    display_sample_images(X_test, y_test, y_pred_classes, correct=False)

    model.save('cnn_rotated_mnist.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN on the Rotated MNIST dataset.') # Argument parser for command line options
    
    parser.add_argument('--data_path', type=str, default='Distributed_Data/Algorithms_Project/RotNIST/data/rotated_mnist.npz', 
                        help='Path to the dataset')# Path to the dataset file
    
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')# Batch size for training the model
    
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs for training')# Number of epochs to train the model
    
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate for the optimizer') # Learning rate for the optimizer
    
    args = parser.parse_args() # Parse the command line arguments
    
    main(args) # Run the main function with parsed arguments
