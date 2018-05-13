import keras
import numpy as np
import matplotlib.pyplot as plt
import utilities
import os
import pickle

import seaborn as sns
current_palette = sns.color_palette('deep')
sns.set_style("whitegrid")
sns.set_palette(current_palette)


class MetricsHandler(keras.callbacks.Callback):
    def __init__(self, valid, output_folder, model_name):
        self.valid = valid
        self.output_folder = output_folder
        self.predictions_folder = output_folder + "prediction_history"
        self.model_name = model_name
        self.predictions = []
        self.accuracy = Accuracy(output_folder, model_name)
        self.spearmanCorrelation = SpearmanCorrelation(output_folder,
                                                       model_name)

    def on_epoch_end(self, epoch, logs={}):
        x_val, targets = self.valid["images"], self.valid["steers"]
        predictions = self.model.predict(x_val)
        self.predictions.append(predictions)

        self.accuracy.eval(predictions, targets)
        self.spearmanCorrelation.eval(predictions, targets)

    def plot_and_save(self):
        self.accuracy.plot_and_save()
        self.spearmanCorrelation.plot_and_save()
        prediction_histogram = PredictionHistogram(self.model, self.valid,
                                                   self.output_folder,
                                                   self.model_name)
        prediction_histogram.plot_and_save()

        utilities.make_folder(self.predictions_folder)
        with open(os.path.join(self.predictions_folder, self.model_name), 'wb') as fp:
            pickle.dump(self.predictions, fp)


class Accuracy:
    def __init__(self, output_folder, model_name, limit=0.1):
        self.output_folder = output_folder + "accuracy"
        self.model_name = model_name
        self.accuracies = []
        self.limit = limit

    def eval(self, predictions, targets):
        accuracy = np.sum((np.absolute(np.subtract(targets, predictions)) < self.limit).astype(int)) / float(len(predictions))
        self.accuracies.append(accuracy)

    def plot_and_save(self):
        xdata = np.arange(1, len(self.accuracies) + 1, 1)
        plt.figure(1)
        plt.plot(xdata, self.accuracies)

        utilities.make_folder(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, self.model_name))
        plt.close('all')
        with open(os.path.join(self.output_folder, self.model_name), 'wb') as fp:
            pickle.dump(self.accuracies, fp)


class SpearmanCorrelation:
    def __init__(self, output_folder, model_name):
        self.output_folder = output_folder + "spearman_correlation"
        self.model_name = model_name
        self.correlations = []

    def eval(self, predictions, targets):
        correlation = np.sum(np.cos(np.absolute(np.subtract(targets,
                                                            predictions)))) \
            / float(len(predictions))
        self.correlations.append(correlation)

    def plot_and_save(self):
        xdata = np.arange(1, len(self.correlations) + 1, 1)
        plt.figure(1)
        plt.plot(xdata, self.correlations)

        utilities.make_folder(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, self.model_name))
        plt.close('all')
        with open(os.path.join(self.output_folder, self.model_name),
                  'wb') as fp:
            pickle.dump(self.correlations, fp)


class PredictionHistogram():
    def __init__(self, model, valid, output_folder, model_name):
        self.model = model
        self.valid = valid
        self.output_folder = output_folder + "histogram"
        self.model_name = model_name

    def plot_and_save(self):
        x_val, y_true = self.valid["images"], self.valid["steers"]
        y_pred = self.model.predict(x_val)
        plt.figure(2)

        ax1 = plt.subplot(211)
        ax1.set_title('Predicted angles')
        ax1.hist(y_pred, bins=40, color=current_palette[0])

        ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
        ax2.set_title('True angles')
        ax2.hist(y_true, bins=40, color=current_palette[1])

        plt.tight_layout()
        utilities.make_folder(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, self.model_name))
        plt.close('all')
