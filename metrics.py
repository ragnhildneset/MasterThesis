import keras
import numpy as np
import matplotlib.pyplot as plt
import utilities
import os
import pickle


class Accuracy(keras.callbacks.Callback):
    def __init__(self, valid, output_folder, model_name):
        self.valid = valid
        self.output_folder = output_folder
        self.model_name = model_name
        self.accuracies = []
        self.limit = 0.1

    def eval_map(self):
        x_val, y_true = self.valid["images"], self.valid["steers"]
        y_pred = self.model.predict(x_val)
        return np.sum((np.absolute(np.subtract(y_true, y_pred)) < self.limit).astype(int)) / len(y_pred)

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        self.accuracies.append(score)

    def plot_and_save(self):
        xdata = np.arange(1, len(self.accuracies) + 1, 1)
        plt.plot(xdata, self.accuracies)

        utilities.make_folder(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, self.model_name))

        with open(os.path.join(self.output_folder, self.model_name), 'wb') as fp:
            pickle.dump(self.accuracies, fp)

