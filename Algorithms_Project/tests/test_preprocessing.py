import unittest
import numpy as np
from unittest.mock import patch
import os
import sys

# Ensure the module path is included
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RotNIST'))

from preprocessing import extract_data, extract_labels, apply_rotation

class TestPreprocessing(unittest.TestCase):

    @patch('preprocessing.extract_data')
    def test_extract_data(self, mock_extract_data):
        """
        Test the extract_data function to ensure it correctly extracts image data from gzip files.
        """
        # Create mock data
        mock_data = np.random.randint(0, 256, size=(100, 28, 28, 1), dtype=np.uint8)
        mock_extract_data.return_value = mock_data

        # Test the extraction function with a small dataset
        data = extract_data('Distributed_Data/Algorithms_Project/RotNIST/data/t10k-images-idx3-ubyte.gz', 100)
        self.assertEqual(data.shape, (100, 28, 28, 1))
        
        # Check if data is within the expected range
        self.assertTrue(np.all(data >= 0) and np.all(data <= 255), "Data values are not in the expected range [0, 255]")

        # Test with a different size
        mock_data = np.random.randint(0, 256, size=(50, 28, 28, 1), dtype=np.uint8)
        mock_extract_data.return_value = mock_data
        data = extract_data('Distributed_Data/Algorithms_Project/RotNIST/data/train-images-idx3-ubyte.gz', 50)
        self.assertEqual(data.shape, (50, 28, 28, 1))

    @patch('preprocessing.extract_labels')
    def test_extract_labels(self, mock_extract_labels):
        """
        Test the extract_labels function to ensure it correctly extracts label data from gzip files.
        """
        # Create mock labels
        mock_labels = np.random.randint(0, 10, size=(100,), dtype=np.uint8)
        mock_extract_labels.return_value = mock_labels

        # Test the extraction function for labels
        labels = extract_labels('Distributed_Data/Algorithms_Project/RotNIST/data/train-labels-idx1-ubyte.gz', 100)
        self.assertEqual(labels.shape, (100,))
        
        # Check if labels are within the expected range
        self.assertTrue(np.all(labels >= 0) and np.all(labels < 10), "Label values are not in the expected range [0, 9]")

        # Test with a different size
        mock_labels = np.random.randint(0, 10, size=(50,), dtype=np.uint8)
        mock_extract_labels.return_value = mock_labels
        labels = extract_labels('Distributed_Data/Algorithms_Project/RotNIST/data/train-labels-idx1-ubyte.gz', 50)
        self.assertEqual(labels.shape, (50,))

    def test_apply_rotation(self):
        """
        Test the apply_rotation function to ensure it correctly applies random rotations to images.
        """
        # Test the rotation function with an all-zero image
        img = np.zeros((28, 28, 1))
        rotated_img = apply_rotation(img)
        self.assertEqual(rotated_img.shape, (28, 28, 1))
        self.assertTrue(np.all(rotated_img == 0), "Rotated image of zeros is not all zeros")

        # Test the rotation function with a non-zero image
        img = np.random.rand(28, 28, 1) * 255
        rotated_img = apply_rotation(img)
        self.assertEqual(rotated_img.shape, (28, 28, 1))
        self.assertTrue(np.any(rotated_img != img), "Rotated image should be different from the original image")

if __name__ == '__main__':
    unittest.main()
