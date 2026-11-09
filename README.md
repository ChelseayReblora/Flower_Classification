
# Flower Classification Project

This repository contains machine learning models built to classify images of flowers using deep learning. The project utilizes TensorFlow and Keras to train and evaluate two different Convolutional Neural Network (CNN) architectures.

## Project Overview

The goal of this project is to accurately categorize flower images into their respective classes. To achieve this, two different pre-trained model architectures were adapted and trained on a custom flower dataset:

*   **MobileNetV2** (`CW3.ipynb` / `image_classifier.py`): A lightweight and efficient model, using `ImageDataGenerator` for data loading and augmentation.
*   **ResNet-50** (`CW3-2.ipynb`): A deeper residual network utilizing TensorFlow's modern `image_dataset_from_directory` for efficient dataset batching.

## Dataset

This project uses a dataset split into two directories:
*   `flower_photos_train/`: Used for training the models.
*   `flower_photos_test/`: Used for validating and testing model accuracy.

*(Note: The dataset images are excluded from this repository via `.gitignore` due to size constraints.)*

## Setup and Installation

To run these notebooks locally, you will need Python 3.11+ and the following libraries installed:

```bash
pip install tensorflow numpy matplotlib jupyter
```

## Usage

1. Clone this repository to your local machine.
2. Ensure you have the dataset folders (`flower_photos_train` and `flower_photos_test`) placed in the root directory of the project.
3. Open either `CW3.ipynb` or `CW3-2.ipynb` in VS Code or Jupyter Notebook.
4. Select your Python/TensorFlow kernel.
5. Run the cells from top to bottom to load the data, train the model, and view the accuracy graphs.
