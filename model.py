#!/usr/bin/python

import architecture
import utilities
import argparse
import os
import visualisation
import metrics
import data_processing

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

MODEL_DIR = "output/models"
HEAT_MAP_DIR = "output/vis/heat_maps"
ANGLE_VIS_DIR = "output/vis/angles"
METRICS_DIR = "output/metrics/"
MERGED_VIS_DIR = "output/vis/angles_and_heat_maps"
DATASET_DIR = "dataset/"
TENSORBOARD_DIR = "output/logs"
SUMMARY_DIR = "output/summary"

RANDOM_SEED = 0


def visualize(model, valid, dataset_dir, vis_size, model_name, base_model):
    vis_sample = base_model.get_random_batch(valid, dataset_dir, vis_size, random_seed=RANDOM_SEED)
    visualisation.make_and_save_angle_visualization(model, vis_sample, dataset_dir, os.path.join(ANGLE_VIS_DIR, model_name))
    visualisation.make_and_save_visualbackprop_masks(model, vis_sample, os.path.join(HEAT_MAP_DIR, model_name))
    visualisation.create_merged_angles_and_heat_maps(MERGED_VIS_DIR, HEAT_MAP_DIR, ANGLE_VIS_DIR, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network to autonomously drive a virtual car. Example syntax:\n\npython model.py -d udacity_dataset -m model.h5')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log and images.')
    parser.add_argument('--model-name', '-m', dest='model_name', type=str, required=True, help='Required string: Name of model e.g model.h5.')
    parser.add_argument('--architecture', '-a', dest='architecture', type=str, required=False, default="Bojarski", help='Nameg of the architecture to be used.')
    parser.add_argument('--augmentation', '-au', dest='augmentation', type=bool, required=False, default=True, help='Use augmentation on training set.')
    parser.add_argument('--gpu-batch-size', '-g', dest='gpu_batch_size', type=int, required=False, default=64, help='Optional integer: Image batch size that fits in VRAM. Default 512.')
    parser.add_argument('--visualization-size', '-vs', dest='vis_size', type=int, required=False, default=50, help='The number of images to be selected for visualisation.')
    parser.add_argument('--epochs', '-e', dest='epochs', type=int, required=False, default=15, help='The number of epochs')
    parser.add_argument('--test-set-name', '-t', dest='test_set_name', type=str, required=False, default='track4*.csv', help='Name of the test set to be used.')
    args = parser.parse_args()

    dataset_path = os.path.join(DATASET_DIR, args.dataset_directory)

    train, valid = utilities.get_dataset_from_folder(dataset_path,
                                                     args.test_set_name)

    if args.augmentation:
        train = data_processing.upsample_large_angles(train)

    base_model = architecture.get_model(args.architecture,
                                        include_speed=False)
    model = base_model.get_model()

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

    valid_processed = base_model.get_random_batch(valid, dataset_path,
                                                  valid.shape[0])

    metrics_handler = metrics.MetricsHandler(valid_processed,
                                                      METRICS_DIR,
                                                      args.model_name)

    model.fit_generator(
        generator=base_model.get_batch_generator(train,
                                                 dataset_path,
                                                 args.gpu_batch_size,
                                                 augmentation=args.augmentation),
        steps_per_epoch=len(train) // args.gpu_batch_size,
        epochs=args.epochs,
        callbacks=[tensorboard, checkpoint, earlyStopping, metrics_handler],
        validation_data=base_model.get_batch_generator(valid,
                                                       dataset_path,
                                                       args.gpu_batch_size,
                                                       augmentation=False),
        validation_steps=(len(valid) // args.gpu_batch_size),
    )

    metrics_handler.plot_and_save()

    visualize(model, valid, dataset_path, args.vis_size, args.model_name,
              base_model)

    utilities.make_folder(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, args.model_name + '.h5'))
    utilities.write_summary(SUMMARY_DIR, args.model_name, args.dataset_directory, train.shape[0], args.test_set_name, valid.shape[0], args.architecture, args.augmentation)
