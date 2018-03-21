#!/usr/bin/python

import architecture
import utilities
import argparse
import os
import numpy as np
import visualisation
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

MODEL_DIR = "output/models"
HEAT_MAP_DIR = "output/vis/heat_maps"
ANGLE_VIS_DIR = "output/vis/angles"
DATASET_DIR = "dataset/"
TENSORBOARD_DIR = "output/logs"

RANDOM_SEED = 0

base_model = architecture.Bojarski_Model(include_speed=False)

class WithinRangePercentage(keras.callbacks.Callback):
    def __init__(self, valid, dataset_dir):
        self.valid = base_model.get_random_batch(valid, dataset_dir, valid.shape[0])
        self.evals = []

    def eval_map(self):
        x_val, y_true = self.valid["images"], self.valid["steers"]
        y_pred = self.model.predict(x_val)
        return np.sum((np.absolute(np.subtract(y_true,y_pred)) < 0.1).astype(int))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        self.evals.append(score)

    def plot_and_save(self):


def visualize(model, valid, dataset_dir, vis_size, model_name, base_model):
    vis_sample = base_model.get_random_batch(valid, dataset_dir, vis_size, random_seed=RANDOM_SEED)
    visualisation.make_and_save_heat_maps(model, vis_sample, 5, os.path.join(HEAT_MAP_DIR, model_name))
    visualisation.make_and_save_angle_visualization(model, vis_sample, dataset_dir, os.path.join(ANGLE_VIS_DIR, model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network to autonomously drive a virtual car. Example syntax:\n\npython model.py -d udacity_dataset -m model.h5')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log and images.')
    parser.add_argument('--model-name', '-m', dest='model_name', type=str, required=True, help='Required string: Name of model e.g model.h5.')
    parser.add_argument('--gpu-batch-size', '-g', dest='gpu_batch_size', type=int, required=False, default=512, help='Optional integer: Image batch size that fits in VRAM. Default 512.')
    parser.add_argument('--test-size', '-ts', dest='test_size', type=int, required=False, default=0.2, help='The fraction of samples used for testing.')
    parser.add_argument('--visualization-size', '-vs', dest='vis_size', type=int, required=False, default=50, help='The number of images to be selected for visualisation.')
    parser.add_argument('--epochs', '-e', dest='epochs', type=int, required=False, default=15, help='The number of images to be selected for visualisation.')
    args = parser.parse_args()

    dataset_path = os.path.join(DATASET_DIR, args.dataset_directory)

    train, valid = utilities.get_dataset_from_folder(dataset_path,
                                                     'first_*.csv')
    model = base_model.get_model()  # initialize neural network model that will be iteratively trained in batches

    # Callbacks
    tensorboard = TensorBoard(log_dir=os.path.join(TENSORBOARD_DIR,
                              args.model_name),
                              histogram_freq=0,
                              write_graph=True, write_images=False)
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR,
                                 args.model_name +
                                 '-{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='val_loss', verbose=0,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto', period=500)

    earlyStopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0, patience=150, verbose=0,
                                  mode='auto')

    withinRangeEval = WithinRangePercentage(valid, dataset_path)

    model.fit_generator(
        generator=base_model.get_batch_generator(train,
                                                 dataset_path,
                                                 args.gpu_batch_size),
        steps_per_epoch=len(train) // args.gpu_batch_size,
        epochs=args.epochs,
        callbacks=[tensorboard, checkpoint, earlyStopping, withinRangeEval],
        validation_data=base_model.get_batch_generator(valid,
                                                       dataset_path,
                                                       args.gpu_batch_size),
        validation_steps=(len(valid) // args.gpu_batch_size),
    )
    print(withinRangeEval.evals)

    visualize(model, valid, dataset_path, args.vis_size, args.model_name,
              base_model)

    utilities.make_folder(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, args.model_name + '.h5'))
