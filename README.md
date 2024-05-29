## Algorithm Purpose
The primary objective of this algorithm is two-fold. First, it processes the original labeled MNIST dataset by applying random rotations to each individual image, which is essential for creating a more realistic neural network model since handwritten digits in real-world scenarios can appear in various orientations. Second, the algorithm evaluates the efficiency gains achieved through parallel processing during the rotation task for the normal sets and rotation and noise for the challenge set, demonstrating how parallelization can significantly expedite the process while maintaining consistent results, thereby improving computational performance.

## Hyperparameters

This algorithm allows users to specify various hyperparameters to fine-tune the performance and behavior of the model. The following hyperparameters can be adjusted:

- **Rotation Angle Range**: Defines the range of angles (e.g., -45 to 45 degrees) within which random rotations are applied to the MNIST images.
- **Batch Size**: Determines the number of images processed in each batch during training, affecting memory usage and training speed.
- **Number of Epochs**: Specifies the number of times the entire training dataset is passed through the model, influencing model accuracy and overfitting.
- **Learning Rate**: Controls the step size during gradient descent, impacting the convergence speed and stability of the training process.
- **Dropout Rate**: Sets the probability of dropping out units in the network to prevent overfitting, enhancing the model's generalization capabilities.
- **Optimizer**: Allows selection of different optimization algorithms (e.g., Adam, SGD) for training the neural network, affecting the efficiency of the learning process.

These hyperparameters can be customized to suit different datasets and training requirements, ensuring flexibility and adaptability of the algorithm to various use cases.

## Background
### History

The MNIST dataset, created by Yann LeCun and others, is a benchmark dataset widely used for training and testing in the field of machine learning. It consists of 70,000 images of handwritten digits (0-9), each sized 28x28 pixels. The dataset has been instrumental in the development of various machine learning algorithms, serving as a standard for evaluating the performance of new models.

### Variations

Over time, numerous variations of the MNIST dataset have been created to challenge and extend the capabilities of machine learning models.

## Pseudo code

### Pseudocode for preprocessing.py

#### Extract and normalize image data
FUNCTION extract_data(filename, num)
    READ and process image data from gzip file
    RETURN normalized data

#### Extract labels from a file
FUNCTION extract_labels(filename, num)
    READ and process labels from gzip file
    RETURN labels

#### Apply random rotations to images in parallel
FUNCTION apply_random_rotations(images)
    APPLY rotation to images using multiple CPUs
    RETURN rotated images

#### Apply challenging transformations to images in parallel
FUNCTION apply_challenging_transformations_parallel(images)
    APPLY complex transformations using multiple CPUs
    RETURN transformed images

#### Prepare MNIST data by extracting, processing, and saving
FUNCTION prepare_MNIST_data()
    EXTRACT training and testing data and labels
    APPLY random rotations to training and testing images
    CREATE challenge dataset with additional transformations
    SPLIT training data into training and validation sets
    SAVE processed data

### Pseudocode for main script

#### Load the data from a file
FUNCTION load_data(filepath)
    LOAD data from npz file
    RETURN X_train, y_train, X_dev, y_dev, X_test, y_test, X_challenge, y_challenge

#### Define the CNN model architecture
FUNCTION create_cnn_model(input_shape, num_classes)
    CREATE Sequential model with layers:
        - Conv2D, BatchNormalization, MaxPooling2D, Dropout
        - Conv2D, BatchNormalization, MaxPooling2D, Dropout
        - Conv2D, BatchNormalization, MaxPooling2D, Dropout
        - Flatten, Dense, BatchNormalization, Dropout
        - Dense (output layer)
    RETURN model

#### Display sample images and their predictions
FUNCTION display_sample_images(images, labels, predictions, sample_size, correct)
    DISPLAY sample images with true and predicted labels

#### Main function to load data, create model, train, evaluate, and save model
FUNCTION main(args)
    LOAD data
    SET input shape and number of classes
    CREATE and compile the CNN model
    CONVERT data to TensorFlow datasets
    TRAIN the model with training and validation datasets
    EVALUATE the model on test and challenge datasets
    PREDICT on the test dataset
    DISPLAY correctly and incorrectly classified images
    SAVE the trained model


## Visualization or animation of algorithm steps or results

Extracting Distributed_Data/Algorithms_Project/RotNIST/data/train-images-idx3-ubyte.gz
Extracting Distributed_Data/Algorithms_Project/RotNIST/data/train-labels-idx1-ubyte.gz
Extracting Distributed_Data/Algorithms_Project/RotNIST/data/t10k-images-idx3-ubyte.gz
Extracting Distributed_Data/Algorithms_Project/RotNIST/data/t10k-labels-idx1-ubyte.gz
Using 10 CPUs on Darwin 23.5.0

Process ID: 42568 is processing an image
Process ID: 42579 is processing an image
Process ID: 42580 is processing an image
Process ID: 42585 is processing an image

Process ID: 42579 is processing a challenge image
Process ID: 42580 is processing a challenge image
Process ID: 42582 is processing a challenge image
Process ID: 42576 is processing a challenge image

## Benchmark Results
Over the versions in which I applied parallelization to the preprocessing and where I did not, these are the average time differences which parallelization saved:

2.180572032928467 seconds
2.383368968963623 seconds
2.339139699935913 seconds
2.054847240447998 seconds
2.293233871459961 seconds

Granted, these results were pulled from the rotation of only 10000 images. Since these rotations are random, duplicate initial images could be used to create multiple copies of the same handwritten numbers with different rotations and noise. While the process only saves ~2.2 seconds over a non-parallel method, over a larger training set this number can see drastic increases. 

## Lessons Learned

I think the largest lesson I learned is that not everything can be multiprocessed. A lot of the time I spent on this project was around trying to get the convolutional neural network to produce a result that was sped up with multiprocessing. I tried numerous different approaches including: a multi-TPU strategy which did not result in a product that could run, performing a mini-batch gradient descent with multiprocessing which degraded the run-time significantly with poorer results, and trying it with stochastic instead of mini-batch. 

None of these resulted in anything that produced a fruitful outcome. So instead of producing a neural network that was erroneous or painfully slow, I applied parallelization to different parts of the preprocessing in order to see how the overall process of a neural network could be sped up. 

Additionally, I have had zero exposure to a lot of the coding proficiencies required for the assignment, speficially relating to the .gitignore file. A lot of the time for this project was spent understanding how to meet the requirements of the project as best as I could.

## Unit-testing strategy

In preprocessing:
Test the extract_data function to ensure it correctly extracts image data from gzip files.
Test the extract_labels function to ensure it correctly extracts label data from gzip files.
Test the apply_rotation function to ensure it correctly applies random rotations to images.

In CNN:
Test the data loading function to ensure it correctly loads and processes the dataset.
Test the CNN model creation to ensure it is correctly configured.
Test the training process of the CNN model with a small subset of data.
Test the prediction capability of the CNN model with a small subset of data.
