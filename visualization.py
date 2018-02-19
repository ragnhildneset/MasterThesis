import data_processing
import matplotlib.pyplot as plt
from vis.visualization import visualize_cam, overlay
from keras.models import load_model
import cv2
from data_processing import *


def heat_map(model_path, data_dir):
    model = load_model(model_path)
    data = data_processing.batch_preprocess(data_dir, measurement_range=(0, 10))

    plt.figure()
    for i, img in enumerate([data["features"][0], data["features"][1]]):
        grads = visualize_cam(model, 5, filter_indices=20, seed_input=img)
        img = un_normalize_color(img)
        plt.imshow(overlay(grads, cv2.convertScaleAbs(img)))

    plt.show()
