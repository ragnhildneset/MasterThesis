import matplotlib 
matplotlib.use("Agg")

import cv2
import utilities
import os
import data_processing
import cv2

from vis.visualization import visualize_cam, overlay



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


def make_and_save_heat_maps(model, samples, layer, output_folder):
    utilities.make_folder(output_folder)

    matplotlib.pyplot.figure()
    for i, image in enumerate(samples["images"]):
        grads = visualize_cam(model, layer, filter_indices=20, seed_input=image)

        display_image = data_processing.un_normalize_color(image)
        matplotlib.pyplot.imshow(overlay(grads, cv2.convertScaleAbs(display_image)))

        figure_name = samples["image_names"][i].replace("/", "_")
        matplotlib.pyplot.savefig(os.path.join(output_folder, figure_name))
