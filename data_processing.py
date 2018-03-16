import numpy as np
import cv2
import utilities


def batch_generator(samples, dataset_path, batch_size, img_size=(67, 320),
                    include_angles=True, include_speed=True, nof_outputs=2):

    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, nof_outputs))

    while True:
        i = 0
        for index in np.random.permutation(samples.shape[0]):
            speed = samples.iloc[index, 0]
            angle = samples.iloc[index, 1]

            image_path = samples.iloc[index, 2]
            image = utilities.load_image(dataset_path, image_path)

            if np.random.rand() < 0.6:
                image, angle = flip(image, angle)

            images[i] = preprocess(image, img_size)
            steers[i] = [angle, speed] if (include_angles and include_speed) \
                else ([speed] if include_speed else [angle])

            i += 1
            if i == batch_size:
                break
        yield images, steers


def random_batch(samples, dataset_path, batch_size, img_size=(67, 320),
                 random_seed=None, include_angles=True, include_speed=True,
                 nof_outputs=2):
    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, nof_outputs))
    image_names = []

    np.random.seed(random_seed)
    for i, index in enumerate(np.random.choice(samples.shape[0], batch_size)):
        speed = samples.iloc[index, 0]
        angle = samples.iloc[index, 1]
        image_name = samples.iloc[index, 2]

        image_path = samples.iloc[index, 2]
        image = utilities.load_image(dataset_path, image_path)

        images[i] = preprocess(image, img_size)
        steers[i] = [angle, speed] if (include_angles and include_speed) \
            else ([speed] if include_speed else [angle])
        image_names.append(image_name)

    np.random.seed(None)
    return {"images": images, "steers": steers, "image_names": image_names}


def flip(image, angle):
    image = cv2.flip(image, flipCode=1)
    angle = angle * -1
    return image, angle


def preprocess(image, img_size):
    reduced_image = reduce_resolution(image, img_size[0], img_size[1])
    normalized_image = normalize_color(reduced_image)
    return normalized_image


def normalize_color(image_matrix):
    return image_matrix / float(255) - 0.5


def un_normalize_color(image_matrix):
    return cv2.convertScaleAbs(image_matrix + 0.5, alpha=255)


def reduce_resolution(image, height, width):
    return cv2.resize(image, (width, height))
