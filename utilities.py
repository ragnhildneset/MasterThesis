import os
import pandas as pd
import glob
import cv2


# get dataset from folder with multiple csv files
def get_dataset_from_folder(dataset_dir, test_set_regex):
    all_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    test_set = glob.glob(os.path.join(dataset_dir, test_set_regex))
    training_set = [x for x in all_files if x not in test_set]
    training_dfs = (pd.read_csv(f) for f in training_set)
    test_dfs = (pd.read_csv(f) for f in test_set)
    return pd.concat(training_dfs,
                     ignore_index=True), pd.concat(test_dfs, ignore_index=True)


# utility function to display cv2 image
def displayCV2(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_folder(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


def load_image(dataset_path, image_file):
    return cv2.imread(os.path.join(dataset_path, image_file))


def write_summary(summary_dir, model_name, dataset_dir, training_size,
                  test_set_name, test_size, architecture, augmentation):
    make_folder(summary_dir)
    f = open(os.path.join(summary_dir, model_name), 'w')
    f.write("Model name: %s \n" % (model_name))
    f.write("Dataset folder:  %s , containing a total of %s elements. \n" %
            (dataset_dir, test_size + training_size))
    f.write("Test set name: %s, containing %s elements. \n" % (test_set_name,
                                                               test_size))
    f.write("Architecture: %s \n" % (architecture))
    f.write("Augmentation: %s \n" % (augmentation))
