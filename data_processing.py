import numpy as np
import cv2
import utilities
import random
import math


def batch_generator(samples, dataset_path, batch_size, img_size=(67, 320),
                    include_angles=True, include_speed=True, nof_outputs=2):

    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, nof_outputs))

    random_erasing = RandomErasing()
    while True:
        i = 0
        for index in np.random.permutation(samples.shape[0]):
            speed = samples.iloc[index, 0]
            angle = samples.iloc[index, 1]

            image_path = samples.iloc[index, 2]
            image = utilities.load_image(dataset_path, image_path)

            # Augmentation
            if np.random.rand() < 0.6:
                image, angle = flip(image, angle)
            image = random_brightness(image)
            if np.random.rand() < 0.6:
                image = random_erasing(image)

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
    reduced_image = reduce_resolution_and_crop_top(image, img_size[0], img_size[1])
    normalized_image = normalize_color(reduced_image)
    return normalized_image


def normalize_color(image_matrix):
    return image_matrix / float(255) - 0.5


def un_normalize_color(image_matrix):
    return cv2.convertScaleAbs(image_matrix + 0.5, alpha=255)


def reduce_resolution(image, height, width):
    return cv2.resize(image, (width, height))


def random_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

class RandomErasing(object):
    def __init__(self, sl=0.02, sh=0.4, r1=0.3, mean=[127, 127, 127]):
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)

                img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                return img

        return img

def reduce_resolution_and_crop_top(image, height, width):
    cropped_top_offset = int(image.shape[0] - (image.shape[0] * 0.9))
    cropped = image[cropped_top_offset:cropped_top_offset + image.shape[0],
                    0:image.shape[1]]
    return reduce_resolution(cropped, height, width)
