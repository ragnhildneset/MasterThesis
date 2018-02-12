import architecture
import utilities
import argparse
import data_processing

LABEL_FILE_NAME = 'labels.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network to autonomously drive a virtual car. Example syntax:\n\npython model.py -d udacity_dataset -m model.h5')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log and images.')
    parser.add_argument('--model-path', '-m', dest='model_path', type=str, required=True, help='Required string: Name of model e.g model.h5.')
    parser.add_argument('--cpu-batch-size', '-c', dest='cpu_batch_size', type=int, required=False, default=1000, help='Optional integer: Image batch size that fits in system RAM. Default 1000.')
    parser.add_argument('--gpu-batch-size', '-g', dest='gpu_batch_size', type=int, required=False, default=512, help='Optional integer: Image batch size that fits in VRAM. Default 512.')
    parser.add_argument('--randomize', '-r', dest='randomize', type=bool, required=False, default=False, help='Optional boolean: Randomize and overwrite driving log. Default False.')
    args = parser.parse_args()

    if args.randomize:
        dataset_log_path = utilities.get_driving_log_path(args.dataset_directory)
        print("Randomizing dataset at", dataset_log_path)
        utilities.randomize_dataset_csv(dataset_log_path)

    measurement_index = 0  # index of measurements in dataset
    dataset_log = utilities.get_dataset_from_csv(args.dataset_directory)
    dataset_size = dataset_log.shape[0]
    # use first 20% of dataset for validation
    validation_batch_size = int(0.2 * dataset_size)

    validation_set = data_processing.batch_preprocess(args.dataset_directory, LABEL_FILE_NAME, measurement_range=(measurement_index, validation_batch_size))

    X_valid = validation_set['features']
    Y_valid = validation_set['labels']

    measurement_index = validation_batch_size  # update measurement index to the end of the validation set
    model = architecture.model()  # initialize neural network model that will be iteratively trained in batches

    while measurement_index < dataset_size:
        end_index = measurement_index + args.cpu_batch_size
        if end_index < dataset_size:
            print("Pre-processing from index", measurement_index, "to index", end_index)
            preprocessed_batch = data_processing.batch_preprocess(args.dataset_directory, LABEL_FILE_NAME, measurement_range=(measurement_index, end_index))
        else:
            print("Pre-processing from index", measurement_index, "to index", dataset_size)
            preprocessed_batch = data_processing.batch_preprocess(args.dataset_directory, LABEL_FILE_NAME, measurement_range=(measurement_index, None))

        X_batch = preprocessed_batch['features']
        Y_batch = preprocessed_batch['labels']

        print("Done preprocessing.")
        print("Features data shape:", X_batch.shape)
        print("Labels data shape:", Y_batch.shape)

        model.fit(X_batch, Y_batch, validation_data=(X_valid, Y_valid), shuffle=True, epochs=15, batch_size=args.gpu_batch_size)
        measurement_index += args.cpu_batch_size
    model.save(args.model_path)
