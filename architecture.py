from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout
import data_processing

class Model:
    def __init__(self, include_angles=True, include_speed=True):
        self.ANGLES = include_angles
        self.SPEED = include_speed
        self.NOF_OUTPUTS = 2 if (self.SPEED and self.ANGLES) else 1
        self.CONV_LAYERS = []

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

    def get_conv_layers(self):
        return self.CONV_LAYERS


class Bojarski_Model(Model):
    def get_model(self):
        self.CONV_LAYERS = range(2, 6 + 1)
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 3)))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.NOF_OUTPUTS))
        model.compile(optimizer="adam", loss="mse", metrics=['accuracy', 'mae'])
        return model

class Bojarski_Model2FC(Model):
    def get_model(self):
        self.CONV_LAYERS = range(2, 6 + 1)
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 3)))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.NOF_OUTPUTS))
        model.compile(optimizer="adam", loss="mse", metrics=['accuracy', 'mae'])
        return model

class Simplified_Bojarski_Model(Model):
    def get_model(self):
        self.CONV_LAYERS = range(2, 5 + 1)
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 3)))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(70, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.NOF_OUTPUTS))
        model.compile(optimizer="adam", loss="mse", metrics=['accuracy', 'mae'])
        return model


class Very_Simplified_Bojarski_Model(Model):
    def get_model(self):
        self.CONV_LAYERS = range(2, 4 + 1)
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 3)))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(35, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.NOF_OUTPUTS))
        model.compile(optimizer="adam", loss="mse", metrics=['accuracy', 'mae'])
        return model


def get_model(name, include_speed=False):
    return {
        'Bojarski': Bojarski_Model(include_speed=include_speed),
        'Bojarski2FC': Bojarski_Model2FC(include_speed=include_speed),
        'Simplified_Bojarski': Simplified_Bojarski_Model(include_speed=include_speed),
        'Very_Simplified_Bojarski' : Very_Simplified_Bojarski_Model(include_speed=include_speed)
    }[name]
