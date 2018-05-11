import matplotlib
matplotlib.use("Agg")  # Needed for docker container with no screen

import cv2
import utilities
import os
import glob
import data_processing

from PIL import Image

import numpy as np
from visual_backprop import VisualBackprop


SEABORN_RED = (82, 78, 196)
SEABORN_GREEN = (104, 168, 85)
SEABORN_BLUE = (176, 114, 76)


def process_img_for_angle_visualization(img, angle, pred_angle, frame):
    font = cv2.FONT_HERSHEY_COMPLEX

    img = cv2.resize(img, (864, 648), interpolation=cv2.INTER_CUBIC)

    h, w = img.shape[0:2]

    # add black rectangle behind text for readability
    img = cv2.rectangle(img, (0, 0), (455, 95), (0, 0, 0), -1)

    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(30, 20),
                fontFace=font, fontScale=0.8, color=SEABORN_BLUE, thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(30, 50), fontFace=font,
                fontScale=0.8, color=SEABORN_GREEN, thickness=1)
    cv2.putText(img, 'predicted angle: ' + str(pred_angle), org=(30, 80),
                fontFace=font, fontScale=0.8, color=SEABORN_RED, thickness=1)

    # apply a line representing the steering angle
    cv2.line(img, (int(w/2), int(h)), (int(w/2-angle*w/4), int(h/2)),
             SEABORN_GREEN, thickness=5)

    if pred_angle is not None:
        cv2.line(img, (int(w/2), int(h)), (int(w/2-pred_angle*w/4), int(h/2)),
                 SEABORN_RED, thickness=3)
    return img


def make_and_save_angle_visualization(model, samples, dataset_dir, output_folder):
    predictions = model.predict(samples["images"])
    utilities.make_folder(output_folder)
    for i, image in enumerate(samples["images"]):
        angle = samples["steers"][i, 0]
        pred_angle = predictions[i, 0]

        display_image = utilities.load_image(dataset_dir,  samples["image_names"][i])
        visualized_image = process_img_for_angle_visualization(display_image, angle, pred_angle, i)

        figure_name = samples["image_names"][i].replace("/", "_")
        cv2.imwrite(os.path.join(output_folder, figure_name), visualized_image)


def make_and_save_visualbackprop_masks(model, samples, output_folder):
    output_folder_name = output_folder
    visual_backprop = VisualBackprop(model)

    utilities.make_folder(output_folder_name)
    matplotlib.pyplot.figure(figsize=(9, 9))
    f, axarr = matplotlib.pyplot.subplots(3, 1)
    for i, image in enumerate(samples["images"]):
        # get mask and get x and y dimension
        mask = visual_backprop.get_mask(image)[:, :, 0]

        # values for colormap normalization
        vmin = np.min(mask)
        vmax = np.percentile(mask, 99)

        display_image = data_processing.un_normalize_color(image)

        # Regular image
        axarr[0].axis('off')
        axarr[0].imshow(display_image)

        # Heatmap overlay
        axarr[1].axis('off')
        axarr[1].imshow(display_image)
        axarr[1].imshow(mask, alpha=.6, cmap='jet', vmin=vmin, vmax=vmax)

        # Black and white mask
        axarr[2].axis('off')
        axarr[2].imshow(mask, cmap='gray', vmin=vmin, vmax=vmax)

        figure_name = samples["image_names"][i].replace("/", "_")
        matplotlib.pyplot.savefig(
            os.path.join(output_folder_name, figure_name))
    matplotlib.pyplot.close('all')


def create_merged_angles_and_heat_maps(merged_dir, heat_map_dir, angle_dir,
                                       model_name):
    utilities.make_folder(os.path.join(merged_dir, model_name))
    heat_maps_dir = os.path.join(heat_map_dir, model_name)
    angles_dir = os.path.join(angle_dir, model_name)

    pngs = glob.glob(os.path.join(
                       os.path.join(heat_map_dir, model_name),
                       "*.png"))
    jpgs = glob.glob(os.path.join(
                       os.path.join(heat_map_dir, model_name),
                       "*.jpg"))

    images = pngs + jpgs
    images = [os.path.basename(os.path.normpath(path)) for path in images]

    for image in images:
        heat_map = Image.open(os.path.join(heat_maps_dir, image))
        angles = Image.open(os.path.join(angles_dir, image))

        width = heat_map.size[0] + angles.size[0]
        height = max(heat_map.size[1], angles.size[1])

        new_image = Image.new('RGB', (width, height))

        new_image.paste(heat_map, (0, 0))
        new_image.paste(angles, (heat_map.size[0], 0))

        new_image.save(os.path.join(
                       os.path.join(merged_dir, model_name), image))
