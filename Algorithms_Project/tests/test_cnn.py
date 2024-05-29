import unittest
import numpy as np
import sys
import os

# Add the path to the CNN module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RotNIST'))

from CNN import create_cnn_model, load_data

class TestCNN(unittest.TestCase):

    def test_load_data(self):
        """
        Test the data loading function to ensure it correctly loads and processes the dataset.
        """
        X_train, y_train, X_dev, y_dev, X_test, y_test, X_challenge, y_challenge = load_data('Distributed_Data/Algorithms_Project/RotNIST/data/rotated_mnist.npz')
        
        # Validate the shapes of the loaded data
        self.assertEqual(X_train.shape, (54000, 28, 28, 1))
        self.assertEqual(y_train.shape, (54000,))
        self.assertEqual(X_dev.shape, (6000, 28, 28, 1))
        self.assertEqual(y_dev.shape, (6000,))
        self.assertEqual(X_test.shape, (10000, 28, 28, 1))
        self.assertEqual(y_test.shape, (10000,))
        self.assertEqual(X_challenge.shape, (2000, 28, 28, 1))  # Adjusted to the correct shape
        self.assertEqual(y_challenge.shape, (2000,))  # Adjusted to the correct shape

        # Print min and max values of datasets for debugging
        print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
        print(f"X_dev min: {X_dev.min()}, max: {X_dev.max()}")
        print(f"X_test min: {X_test.min()}, max: {X_test.max()}")
        print(f"X_challenge min: {X_challenge.min()}, max: {X_challenge.max()}")

        # Check if data is normalized to [0, 1]
        self.assertTrue(np.all(X_train >= 0) and np.all(X_train <= 1), "X_train values are not in the expected range [0, 1]")
        self.assertTrue(np.all(X_dev >= 0) and np.all(X_dev <= 1), "X_dev values are not in the expected range [0, 1]")
        self.assertTrue(np.all(X_test >= 0) and np.all(X_test <= 1), "X_test values are not in the expected range [0, 1]")
        self.assertTrue(np.all(X_challenge >= 0) and np.all(X_challenge <= 1), "X_challenge values are not in the expected range [0, 1]")

        # Validate label ranges
        self.assertTrue(np.all(y_train >= 0) and np.all(y_train < 10))
        self.assertTrue(np.all(y_dev >= 0) and np.all(y_dev < 10))
        self.assertTrue(np.all(y_test >= 0) and np.all(y_test < 10))
        self.assertTrue(np.all(y_challenge >= 0) and np.all(y_challenge < 10))

    def test_create_cnn_model(self):
        """
        Test the CNN model creation to ensure it is correctly configured.
        """
        model = create_cnn_model((28, 28, 1), 10)
        
        # Validate input and output shapes of the model
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, 10))

        # Test model compilation
        try:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            self.fail(f"Model compilation failed with exception: {e}")

    def test_model_training(self):
        """
        Test the training process of the CNN model with a small subset of data.
        """
        # Load a small subset of data for training test
        X_train, y_train, _, _, _, _, _, _ = load_data('Distributed_Data/Algorithms_Project/RotNIST/data/rotated_mnist.npz')
        X_train, y_train = X_train[:100], y_train[:100]  # Use a small subset for quick test

        model = create_cnn_model((28, 28, 1), 10)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        try:
            model.fit(X_train, y_train, epochs=1, batch_size=10)
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_model_prediction(self):
        """
        Test the prediction capability of the CNN model with a small subset of data.
        """
        # Load a small subset of data for prediction test
        _, _, _, _, X_test, y_test, _, _ = load_data('Distributed_Data/Algorithms_Project/RotNIST/data/rotated_mnist.npz')
        X_test, y_test = X_test[:10], y_test[:10]  # Use a small subset for quick test

        model = create_cnn_model((28, 28, 1), 10)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_test, y_test, epochs=1, batch_size=10)

        try:
            predictions = model.predict(X_test)
            self.assertEqual(predictions.shape, (10, 10))
        except Exception as e:
            self.fail(f"Model prediction failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
