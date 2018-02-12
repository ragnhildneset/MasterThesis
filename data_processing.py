import pandas as pd
import numpy as np

import cv2
import os
import matplotlib.pyplot as plt

dir_path = "vel_test2/Original/"
data_file = "labels.csv"


def read_and_preprocess_data():
    samples = pd.read_csv(os.path.join(dir_path, data_file))
    num_samples = len(samples)
    num_aug_samples = num_samples * 2  # Sample size doubled by flipping each image and corresponding wheel angle.

    Y_train = np.zeros((num_aug_samples, 2))
    X_train = np.zeros((num_aug_samples, 67, 320, 3))
    for sample_index in range(num_samples):
        aug_sample_index = sample_index * 2
        image_file = samples.iloc[sample_index, 2]

        # Read and process image
        image_matrix = cv2.imread(os.path.join(dir_path, image_file))
        reduced_image_matrix = reduce_resolution(image_matrix, 320, 67)
        normalized_image_matrix = normalize_color(reduced_image_matrix)

        # Add to training data
        X_train[aug_sample_index, :, :, :] = normalized_image_matrix  # Image
        Y_train[aug_sample_index, 0] = samples.iloc[sample_index, 0]  # Speed
        Y_train[aug_sample_index, 1] = samples.iloc[sample_index, 1]  # Angle

        # Add flipped training data
        X_train[aug_sample_index + 1, :, :, :] = cv2.flip(normalized_image_matrix, flipCode=1)  # Flipped image
        Y_train[aug_sample_index + 1, 0] = samples.iloc[sample_index, 0]  # Speed
        Y_train[aug_sample_index + 1, 1] = samples.iloc[sample_index, 1] * -1  # Flipped angle

        if False:
            plt.figure(figsize=(15, 5))
            show_image((2, 2, 1), "Normal", cv2.cvtColor(cv2.convertScaleAbs(X_train[aug_sample_index] + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            show_image((2, 2, 2), "Flipped", cv2.cvtColor(cv2.convertScaleAbs(X_train[aug_sample_index + 1] + 0.5, alpha=255), cv2.COLOR_BGR2RGB))
            plt.show()
            plt.close()

    preprocessed_dataset = {'features': X_train, 'labels': Y_train}
    return preprocessed_dataset


def normalize_color(image_matrix):
    return image_matrix / 255 - 0.5


def reduce_resolution(image, width, height):
    return cv2.resize(image, (width, height))

def show_image(location, title, img, width=2, open_new_window=False):
    if open_new_window:
        plt.figure(figsize=(width, 1))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.imshow(img)


read_and_preprocess_data()