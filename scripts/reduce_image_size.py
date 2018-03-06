import os
import utilities
import data_processing
import cv2
from shutil import copyfile

#
# Copy all content in dataset_path folder to a new folder with reduced image sizes
#
dataset_path = "../dataset/example_data"
new_dataset_path = dataset_path + "_reduced"

utilities.make_folder(new_dataset_path)

dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
csvs = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

for image_dir in dirs:
    current_dir = os.path.join(dataset_path, image_dir)
    new_current_dir = os.path.join(new_dataset_path, image_dir)

    utilities.make_folder(new_current_dir)
    image_names = os.listdir(current_dir)
    for image_name in image_names:
        print(current_dir, image_name)
        image = utilities.load_image(current_dir, image_name)
        if image is not None:
            reduced_image = data_processing.reduce_resolution(image, 67, 320)
            cv2.imwrite(os.path.join(new_current_dir, image_name), reduced_image)


for csv in csvs:
    copyfile(os.path.join(dataset_path, csv), os.path.join(new_dataset_path, csv))