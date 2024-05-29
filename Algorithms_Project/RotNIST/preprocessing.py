import gzip
import os
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count, current_process
import platform
import time

def extract_data(filename, num):
    """
    Extract images from the specified gzip file into a 3D tensor [image index, y, x].

    Args:
        filename (str): Path to the gzip file containing image data.
        num (int): Number of images to extract.

    Returns:
        numpy.ndarray: 3D tensor containing the extracted and normalized image data.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, 28, 28, 1)  # Reshape into tensor
        data = data / 255.0  # Normalize to [0, 1] range
    return data

def extract_labels(filename, num):
    """
    Extract labels from the specified gzip file into a vector of int64 label IDs.

    Args:
        filename (str): Path to the gzip file containing label data.
        num (int): Number of labels to extract.

    Returns:
        numpy.ndarray: Vector containing the extracted label IDs.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def apply_rotation(img):
    """
    Apply a random rotation to an image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Rotated image.
    """
    process_id = current_process().pid
    print(f"Process ID: {process_id} is processing an image")
    angle = np.random.randint(0, 360)
    return ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0.0)

def apply_challenging_transformations(img):
    """
    Apply more challenging transformations (rotation and noise) to an image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Transformed image.
    """
    process_id = current_process().pid
    print(f"Process ID: {process_id} is processing a challenge image")
    angle = np.random.randint(0, 360)
    rotated_img = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0.0)
    noisy_img = rotated_img + np.random.normal(0, 0.1, rotated_img.shape)
    return np.clip(noisy_img, 0, 1)

def apply_random_rotations(images):
    """
    Apply random rotations to the images in parallel using multiple CPU cores.

    Args:
        images (numpy.ndarray): Array of images to rotate.

    Returns:
        numpy.ndarray: Array of rotated images.
    """
    print(f"Using {cpu_count()} CPUs on {platform.system()} {platform.release()}")
    with Pool(cpu_count()) as pool:
        rotated_images = pool.map(apply_rotation, images)
    return np.array(rotated_images)

def apply_challenging_transformations_parallel(images):
    """
    Apply challenging transformations to the images in parallel using multiple CPU cores.

    Args:
        images (numpy.ndarray): Array of images to transform.

    Returns:
        numpy.ndarray: Array of transformed images.
    """
    print(f"Using {cpu_count()} CPUs on {platform.system()} {platform.release()}")
    with Pool(cpu_count()) as pool:
        transformed_images = pool.map(apply_challenging_transformations, images)
    return np.array(transformed_images)

def apply_random_rotations_sequential(images):
    """
    Apply random rotations to the images sequentially.

    Args:
        images (numpy.ndarray): Array of images to rotate.

    Returns:
        numpy.ndarray: Array of rotated images.
    """
    rotated_images = [apply_rotation(img) for img in images]
    return np.array(rotated_images)

def apply_challenging_transformations_sequential(images):
    """
    Apply challenging transformations to the images sequentially.

    Args:
        images (numpy.ndarray): Array of images to transform.

    Returns:
        numpy.ndarray: Array of transformed images.
    """
    transformed_images = [apply_challenging_transformations(img) for img in images]
    return np.array(transformed_images)

def prepare_MNIST_data(parallel=True):
    """
    Prepare the MNIST dataset by extracting data, applying transformations, and splitting the data.

    Args:
        parallel (bool): Whether to apply transformations in parallel or sequentially.
    """
    # File paths for the MNIST dataset
    train_data_filename = 'Distributed_Data/Algorithms_Project/RotNIST/data/train-images-idx3-ubyte.gz'
    train_labels_filename = 'Distributed_Data/Algorithms_Project/RotNIST/data/train-labels-idx1-ubyte.gz'
    test_data_filename = 'Distributed_Data/Algorithms_Project/RotNIST/data/t10k-images-idx3-ubyte.gz'
    test_labels_filename = 'Distributed_Data/Algorithms_Project/RotNIST/data/t10k-labels-idx1-ubyte.gz'

    # Extract data into numpy arrays
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    if parallel:
        # Apply random rotations to the images in parallel
        train_data = apply_random_rotations(train_data)
        test_data = apply_random_rotations(test_data)

        # Create a challenge dataset
        challenge_data = apply_challenging_transformations_parallel(test_data[:2000])
    else:
        # Apply random rotations to the images sequentially
        train_data = apply_random_rotations_sequential(train_data)
        test_data = apply_random_rotations_sequential(test_data)

        # Create a challenge dataset
        challenge_data = apply_challenging_transformations_sequential(test_data[:2000])

    challenge_labels = test_labels[:2000]

    # Split train data into train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(train_data, train_labels, test_size=0.10, random_state=42)

    # Save the data in a format that can be easily loaded later
    np.savez_compressed('/Users/benhepner/Documents/VSCODE/Distributed_Data/Algorithms_Project/RotNIST/data/rotated_mnist.npz', X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev, X_test=test_data, y_test=test_labels, X_challenge=challenge_data, y_challenge=challenge_labels)

if __name__ == '__main__':
    print("Running with parallel processing")
    start_time = time.time()
    prepare_MNIST_data(parallel=True)
    end_time = time.time()
    print(f"Parallel processing time: {end_time - start_time} seconds")

    print("Running without parallel processing")
    start_time1 = time.time()
    prepare_MNIST_data(parallel=False)
    end_time1 = time.time()
    print(f"Sequential processing time: {end_time1 - start_time1} seconds")

    print(f"Parallel saves: {(end_time1 - start_time1) - (end_time - start_time)} seconds")
