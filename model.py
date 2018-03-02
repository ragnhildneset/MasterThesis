#!/usr/bin/python

import architecture
import utilities
import argparse
import data_processing
import os
import numpy as np
import visualisation

from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

MODEL_DIR = "output/models"
HEAT_MAP_FOLDER = "output/vis/heat_maps"
ANGLE_VIS_FOLDER = "output/vis/angles"


def visualize(model, valid, dataset_dir, vis_size):
    vis_sample = data_processing.random_batch(valid, dataset_dir, vis_size, random_seed=0)
    visualisation.make_and_save_heat_maps(model, vis_sample, 5, HEAT_MAP_FOLDER)
    visualisation.make_and_save_angle_visualization(model, vis_sample, args.dataset_directory, ANGLE_VIS_FOLDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network to autonomously drive a virtual car. Example syntax:\n\npython model.py -d udacity_dataset -m model.h5')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log and images.')
    parser.add_argument('--model-name', '-m', dest='model_name', type=str, required=True, help='Required string: Name of model e.g model.h5.')
    parser.add_argument('--cpu-batch-size', '-c', dest='cpu_batch_size', type=int, required=False, default=1000000, help='Optional integer: Image batch size that fits in system RAM. Default 1000.')
    parser.add_argument('--gpu-batch-size', '-g', dest='gpu_batch_size', type=int, required=False, default=512, help='Optional integer: Image batch size that fits in VRAM. Default 512.')
    parser.add_argument('--randomize', '-r', dest='randomize', type=bool, required=False, default=False, help='Optional boolean: Randomize and overwrite driving log. Default False.')
    parser.add_argument('--tensorboard-dir', '-t', dest='tensorboard_dir', type=str, required=False, default='output/logs', help='The directory in which the tensorboard logs should be saved.')
    parser.add_argument('--test-size', '-ts', dest='test_size', type=int, required=False, default=0.2, help='The fraction of samples used for testing.')
    parser.add_argument('--visualization-size', '-vs', dest='vis_size', type=int, required=False, default=50, help='The number of images to be selected for visualisation.')
    parser.add_argument('--epochs', '-e', dest='epochs', type=int, required=False, default=15, help='The number of images to be selected for visualisation.')
    args = parser.parse_args()

    dataset_log = utilities.get_dataset_from_folder(args.dataset_directory)
    train, valid = train_test_split(dataset_log, test_size=args.test_size, random_state=0)
    model = architecture.bojarski_model()  # initialize neural network model that will be iteratively trained in batches

    tensorboard = TensorBoard(log_dir=args.tensorboard_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

    model.fit_generator(
        generator=data_processing.batch_generator(train, args.dataset_directory, args.gpu_batch_size),
        steps_per_epoch=len(train) // args.gpu_batch_size,
        epochs=args.epochs,
        callbacks=[tensorboard],
        validation_data=data_processing.batch_generator(valid, args.dataset_directory, args.gpu_batch_size),
        validation_steps=(len(train) // args.gpu_batch_size)
    )

    visualize(model, valid, args.dataset_directory, args.vis_size)

    utilities.make_folder(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, args.model_name))


