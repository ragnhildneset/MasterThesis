import numpy as np
import cv2
import utilities
import random
import math
import os


def batch_generator(samples, dataset_path, batch_size, img_size,
                    include_angles=True, include_speed=True, nof_outputs=2,
                    augmentation=False):

    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, nof_outputs))

    while True:
        i = 0
        for index in np.random.permutation(samples.shape[0]):
            sample = samples.iloc[index]

            angle = sample["steering"]
            speed = sample["speed"]
            image = utilities.load_image(dataset_path, sample["center"])

            if augmentation:
                if np.random.rand() < 0.5:
                    left = utilities.load_image(dataset_path, sample["left"])
                    right = utilities.load_image(dataset_path, sample["right"])
                    image, angle = use_side_cameras(left, right, angle)
                if np.random.rand() < 0.6:
                    image, angle = flip(image, angle)
                image = random_hsv_adjustment(image)
                if np.random.rand() < 0.3:
                    image = erasing_spots(image)
                if np.random.rand() < 0.8:
                    image, angle = random_translations(image, angle)

            images[i] = preprocess(image, img_size)
            steers[i] = [angle, speed] if (include_angles and include_speed) \
                else ([speed] if include_speed else [angle])

            i += 1
            if i == batch_size:
                break
        yield images, steers


def random_batch(samples, dataset_path, batch_size, img_size,
                 random_seed=None, include_angles=True, include_speed=True,
                 nof_outputs=2):
    images = np.zeros((batch_size, img_size[0], img_size[1], 3))
    steers = np.zeros((batch_size, nof_outputs))
    image_names = []

    np.random.seed(random_seed)
    for i, index in enumerate(np.random.choice(samples.shape[0], batch_size)):
        sample = samples.iloc[index]

        angle = sample["steering"]
        speed = sample["speed"]
        image = utilities.load_image(dataset_path, sample["center"])
        image_name = sample["center"]

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


def use_side_cameras(left, right, angle):
    if np.random.rand() < 0.6:
        rot_angle = angle + 0.25
        return left, rot_angle
    rot_angle = angle - 0.25
    return right, rot_angle


def random_hsv_adjustment(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)

    random_bright = 0.5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255

    random_color = 0.9 + np.random.uniform()*0.2
    image1[:, :, 0] = image1[:, :, 0] * random_color
    image1[:, :, 0][image1[:, :, 0] > 255] = 255

    random_saturation = 0.5 + np.random.uniform()
    image1[:, :, 1] = image1[:, :, 1] * random_saturation
    image1[:, :, 1][image1[:, :, 1] > 255] = 255

    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def erasing_spots(img, sl=0.02, sh=0.4, r1=0.3):
    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)

            img[x1:x1 + h, y1:y1 + w, 0] = 255
            img[x1:x1 + h, y1:y1 + w, 1] = 255
            img[x1:x1 + h, y1:y1 + w, 2] = 255
            return img

    return img


def reduce_resolution_and_crop_top(image, height, width):
    cropped_top_offset = int(image.shape[0] - (image.shape[0] * 0.65))
    cropped = image[cropped_top_offset:cropped_top_offset + image.shape[0],
                    0:image.shape[1]]
    return reduce_resolution(cropped, height, width)


def brightness_spots(img, sl=0.02, sh=0.4, r1=0.3):
    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img1 = np.array(img1, dtype=np.float64)

        brightness_factor = random.uniform(1.1, 1.5) if random.random() < 0.7 else random.randint(2,5)

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)

            img1[x1:x1 + h, y1:y1 + w, 2] = img1[x1:x1 + h, y1:y1 + w, 2] * brightness_factor
            img1[x1:x1 + h, y1:y1 + w, 2][img1[x1:x1 + h, y1:y1 + w, 2] > 255] = 255

            img1 = img1.astype(np.uint8)
            img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2RGB)

            return img1

    return img


def random_translations(image, angle, min_x=-30, max_x=30, angle_scale=0.0064):
    tr_x = np.random.randint(min_x, max_x)
    tr_angle = angle + tr_x * angle_scale
    tr_angle = min(1, max(-1, tr_angle))
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, 0]])
    rows, cols = image.shape[0], image.shape[1]
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, tr_angle


def upsample_large_angles(samples):
    is_large_angle = samples['steering'].abs() > 0.5
    samples_large = samples[is_large_angle]

    return samples.append([samples_large]*5)
