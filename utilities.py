import os
import pandas as pd
import glob

def get_driving_log_path(image_input_dir):
    assert (os.path.exists(image_input_dir))
    log_file_list = glob.glob(os.path.join(image_input_dir, '*.csv'))
    assert (len(log_file_list) == 1)
    log_path = log_file_list[0]
    return log_path

def randomize_dataset_csv(csv_path):
    driving_log = pd.read_csv(csv_path, header=None)
    driving_log = driving_log.sample(frac=1).reset_index(drop=True)
    print("Overwriting CSV file: ", csv_path)
    driving_log.to_csv(csv_path, header=None, index=False)
    print("Done.")

def get_dataset_from_csv(image_input_dir):
    log_path = get_driving_log_path(image_input_dir)
    print("Reading from CSV log file", log_path)
    driving_log = pd.read_csv(log_path, header=None)
    return driving_log
