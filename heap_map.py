import data_processing
import matplotlib.pyplot as plt
from vis.visualization import visualize_cam, overlay
from keras.models import load_model
import cv2
from data_processing import *
import utilities
import os


def make_and_save_heat_maps(model, sample, layer, figure_folder):
    utilities.make_folder(figure_folder)

    plt.figure()
    for i in range(sample["images"].shape[0]):
        image = sample["images"][i]
        grads = visualize_cam(model, layer, filter_indices=20, seed_input=image)

        display_image = un_normalize_color(image)
        plt.imshow(overlay(grads, cv2.convertScaleAbs(display_image)))

        figure_name = sample["image_names"][i].replace("/", "_")
        plt.savefig(os.path.join(figure_folder, figure_name))
