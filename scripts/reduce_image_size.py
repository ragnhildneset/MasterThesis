import os
import cv2
import sys
from shutil import copyfile

#
# Copy all content in dataset_path folder to a new folder with reduced image sizes
#

dataset_path = sys.argv[1]  # f.i. ../dataset/ordered_training_data - The path to the top folder of the dataset from current dir (../dataset/ordered_training_data)
new_dataset_path = dataset_path + "_reduced"

if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
csvs = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

for image_dir in dirs:
    current_dir = os.path.join(dataset_path, image_dir)
    new_current_dir = os.path.join(new_dataset_path, image_dir)

    if not os.path.exists(new_current_dir):
        os.makedirs(new_current_dir)
    print("Copying and reducing content in ", current_dir)
    image_names = os.listdir(current_dir)
    for image_name in image_names:
        image = cv2.imread(os.path.join(current_dir, image_name))
        if image is not None:
            reduced_image = cv2.resize(image, (320, 67))
            cv2.imwrite(os.path.join(new_current_dir, image_name), reduced_image)


for csv in csvs:
    copyfile(os.path.join(dataset_path, csv), os.path.join(new_dataset_path, csv))