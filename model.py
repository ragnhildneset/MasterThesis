#!/usr/bin/python

import architecture
import utilities
import argparse
import data_processing
import os

from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

MODEL_DIR = "output/models"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network to autonomously drive a virtual car. Example syntax:\n\npython model.py -d udacity_dataset -m model.h5')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log and images.')
    parser.add_argument('--model-name', '-m', dest='model_name', type=str, required=True, help='Required string: Name of model e.g model.h5.')
    parser.add_argument('--cpu-batch-size', '-c', dest='cpu_batch_size', type=int, required=False, default=1000000, help='Optional integer: Image batch size that fits in system RAM. Default 1000.')
    parser.add_argument('--gpu-batch-size', '-g', dest='gpu_batch_size', type=int, required=False, default=512, help='Optional integer: Image batch size that fits in VRAM. Default 512.')
    parser.add_argument('--randomize', '-r', dest='randomize', type=bool, required=False, default=False, help='Optional boolean: Randomize and overwrite driving log. Default False.')
    parser.add_argument('--tensorboard-dir', '-t', dest='tensorboard_dir', type=str, required=False, default='output/logs', help='The directory in which the tensorboard logs should be saved.')
    parser.add_argument('--test-size', '-ts', dest='test_size', type=int, required=False, default=0.2, help='The fraction of samples used for testing.')
    args = parser.parse_args()

    dataset_log = utilities.get_dataset_from_folder(args.dataset_directory)
    train, valid = train_test_split(dataset_log, test_size=args.test_size, random_state=0)

    model = architecture.bojarski_model()  # initialize neural network model that will be iteratively trained in batches

    tensorboard = TensorBoard(log_dir=args.tensorboard_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

    model.fit_generator(
        generator=data_processing.batch_generator(train, args.dataset_directory, args.gpu_batch_size),
        steps_per_epoch=len(train) // args.gpu_batch_size,
        epochs=15,
        callbacks=[tensorboard],
        validation_data=data_processing.batch_generator(valid, args.dataset_directory, args.gpu_batch_size),
        validation_steps=(len(train) // args.gpu_batch_size)
    )

    utilities.make_folder(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, args.model_name))


