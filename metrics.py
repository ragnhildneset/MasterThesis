import keras
import numpy as np
import matplotlib.pyplot as plt
import utilities
import os
import pickle

import seaborn as sns
sns.set(style="darkgrid")

class Accuracy(keras.callbacks.Callback):
    def __init__(self, valid, output_folder, model_name):
        self.valid = valid
        self.output_folder = output_folder
        self.model_name = model_name
        self.accuracies = []
        self.limit = 0.1

    def eval(self):
        x_val, y_true = self.valid["images"], self.valid["steers"]
        y_pred = self.model.predict(x_val)
        return np.sum((np.absolute(np.subtract(y_true, y_pred)) < self.limit).astype(int)) / float(len(y_pred))

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval()
        self.accuracies.append(score)

    def plot_and_save(self):
        xdata = np.arange(1, len(self.accuracies) + 1, 1)
        plt.figure(1)
        plt.plot(xdata, self.accuracies)

        utilities.make_folder(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, self.model_name))
        plt.close('all')
        with open(os.path.join(self.output_folder, self.model_name), 'wb') as fp:
            pickle.dump(self.accuracies, fp)


class PredictionHistogram():
    def __init__(self, model, valid, output_folder, model_name):
        self.model = model
        self.valid = valid
        self.output_folder = output_folder
        self.model_name = model_name

    def plot_and_save(self):
        x_val, y_true = self.valid["images"], self.valid["steers"]
        y_pred = self.model.predict(x_val)
        plt.figure(2)
        f, axarr = plt.subplots(2, 1)
        axarr[0].set_title('Predicted angles')
        axarr[0].hist(y_pred)
        axarr[1].set_title('True angles')
        axarr[1].hist(y_true)

        utilities.make_folder(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, self.model_name))
        plt.close('all')
