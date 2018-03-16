from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D
import data_processing

class Model:
    def __init__(self, include_angles=True, include_speed=True):
        self.ANGLES = include_angles
        self.SPEED = include_speed
        self.NOF_OUTPUTS = 2 if (self.SPEED and self.ANGLES) else 1

    def get_batch_generator(self, data, dataset_path, batch_size,
                            img_size=(67, 320)):
        return data_processing.batch_generator(data, dataset_path,
                                               batch_size, img_size,
                                               include_angles=self.ANGLES,
                                               include_speed=self.SPEED,
                                               nof_outputs=self.NOF_OUTPUTS)

    def get_random_batch(self, data, dataset_path, batch_size,
                         img_size=(67, 320), random_seed=None):
        return data_processing.random_batch(data, dataset_path,
                                            batch_size, img_size,
                                            random_seed=random_seed,
                                            include_angles=self.ANGLES,
                                            include_speed=self.SPEED,
                                            nof_outputs=self.NOF_OUTPUTS)


class Bojarski_Model(Model):
    def get_model(self):

        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 3)))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.NOF_OUTPUTS))
        model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
        return model
