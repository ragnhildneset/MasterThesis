import numpy as np
import cv2
import os


def batch_generator(samples, dataset_path, batch_size, img_size=(67, 320)):
    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, 2))

    while True:
        i = 0
        for index in np.random.permutation(samples.shape[0]):
            speed = samples.iloc[index, 0]
            angle = samples.iloc[index, 1]

            image_path = samples.iloc[index, 2]
            image = load_image(dataset_path, image_path)

            if np.random.rand() < 0.6:
                image, angle = flip(image, angle)

            images[i] = preprocess(image, img_size)
            steers[i] = [speed, angle]

            i += 1
            if i == batch_size:
                break
        yield images, steers


def random_batch(samples, dataset_path, batch_size, img_size=(67, 320), random_seed=None):
    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, 2))
    image_names = []

    np.random.seed(random_seed)
    for i, index in enumerate(np.random.choice(samples.shape[0], batch_size)):
        speed = samples.iloc[index, 0]
        angle = samples.iloc[index, 1]
        image_name = samples.iloc[index, 2]

        image_path = samples.iloc[index, 2]
        image = load_image(dataset_path, image_path)

        images[i] = preprocess(image, img_size)
        steers[i] = [speed, angle]
        image_names.append(image_name)

    np.random.seed(None)
    return {"images": images, "steers": steers, "image_names": image_names}


def flip(image, angle):
    image = cv2.flip(image, flipCode=1)
    angle = angle * -1
    return image, angle


def load_image(dataset_path, image_file):
    return cv2.imread(os.path.join(dataset_path, image_file))


def preprocess(image, img_size):
    reduced_image = reduce_resolution(image, img_size[0], img_size[1])
    normalized_image = normalize_color(reduced_image)
    return normalized_image


def normalize_color(image_matrix):
    return image_matrix / 255 - 0.5


def un_normalize_color(image_matrix):
    return cv2.convertScaleAbs(image_matrix + 0.5, alpha=255)


def reduce_resolution(image, height, width):
    return cv2.resize(image, (width, height))


