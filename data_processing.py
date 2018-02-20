import pandas as pd
import numpy as np

import cv2
import os
import matplotlib.pyplot as plt
import utilities


def batch_generator(dataset_path, batch_size, img_size=(67, 320)):
    samples = utilities.get_dataset_from_folder(dataset_path)
    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, 2))

    while True:
        i = 0
        for index in np.random.permutation(samples.shape[0]):
            speed = samples.iloc[index, 0]
            angle = samples.iloc[index, 1]

            image_path = samples.iloc[index, 2]
            image = load_image(dataset_path, image_path)

            images[i] = preprocess(image)
            steers[i] = [speed, angle]
            i += 1
            if i == batch_size:
                break
        yield images, steers


def load_image(dataset_path, image_file):
    return cv2.imread(os.path.join(dataset_path, image_file))


def preprocess(image, img_size):
    reduced_image = reduce_resolution(image, img_size[0], img_size[1])
    normalized_image = normalize_color(reduced_image)
    return normalized_image


def batch_preprocess(dir_path, img_size=(67, 320), measurement_range=None):
    samples = utilities.get_dataset_from_folder(dir_path)

    measurement_index = measurement_range[0] if measurement_range else 0
    start_index = measurement_index
    max_measurement_index = measurement_range[1] if measurement_range[1] else len(samples)
    assert (measurement_index < max_measurement_index)

    num_measurements = max_measurement_index - measurement_index
    num_aug_measurements = num_measurements * 2  # Sample size doubled by flipping each image and corresponding wheel angle.

    Y_train = np.zeros((num_aug_measurements, 2))
    X_train = np.zeros((num_aug_measurements, img_size[0], img_size[1], 3))

    while measurement_index < max_measurement_index:
        aug_sample_index = (measurement_index - start_index) * 2
        image_file = samples.iloc[measurement_index, 2]

        # Read and process image
        image_matrix = cv2.imread(os.path.join(dir_path, image_file))
        reduced_image_matrix = reduce_resolution(image_matrix, img_size[0], img_size[1])
        normalized_image_matrix = normalize_color(reduced_image_matrix)

        # Add to training data
        X_train[aug_sample_index, :, :, :] = normalized_image_matrix  # Image
        Y_train[aug_sample_index, 0] = samples.iloc[measurement_index, 0]  # Speed
        Y_train[aug_sample_index, 1] = samples.iloc[measurement_index, 1]  # Angle

        # Add flipped training data
        X_train[aug_sample_index + 1, :, :, :] = cv2.flip(normalized_image_matrix, flipCode=1)  # Flipped image
        Y_train[aug_sample_index + 1, 0] = samples.iloc[measurement_index, 0]  # Speed
        Y_train[aug_sample_index + 1, 1] = samples.iloc[measurement_index, 1] * -1  # Flipped angle

        measurement_index += 1

        if False:
            plt.figure(figsize=(15, 5))
            show_image((2, 2, 1), "Normal", X_train[aug_sample_index])
            show_image((2, 2, 2), "Flipped", X_train[aug_sample_index + 1])
            plt.show()
            plt.close()

    preprocessed_dataset = {'features': X_train, 'labels': Y_train}
    return preprocessed_dataset


def normalize_color(image_matrix):
    return image_matrix / 255 - 0.5


def un_normalize_color(image_matrix):
    return cv2.convertScaleAbs(image_matrix + 0.5, alpha=255)


def reduce_resolution(image, height, width):
    return cv2.resize(image, (width, height))


def show_image(location, title, img, width=2, open_new_window=False):
    if open_new_window:
        plt.figure(figsize=(width, 1))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(un_normalize_color(img), cv2.COLOR_BGR2RGB))
