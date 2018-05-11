from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout
from keras import optimizers
import data_processing


class Model:
    def __init__(self, learning_rate, include_angles=True, include_speed=True,
                 img_size=(66, 200)):
        self.ANGLES = include_angles
        self.SPEED = include_speed
        self.NOF_OUTPUTS = 2 if (self.SPEED and self.ANGLES) else 1
        self.IMG_SIZE = img_size
        self.INPUT_SHAPE = (self.IMG_SIZE[0], self.IMG_SIZE[1], 3)
        self.LEARNING_RATE = learning_rate

    def get_batch_generator(self, data, dataset_path, batch_size,
                            augmentation=False):
        return data_processing.batch_generator(data, dataset_path,
                                               batch_size,
                                               img_size=self.IMG_SIZE,
                                               include_angles=self.ANGLES,
                                               include_speed=self.SPEED,
                                               nof_outputs=self.NOF_OUTPUTS,
                                               augmentation=augmentation)

    def get_random_batch(self, data, dataset_path, batch_size,
                         random_seed=None):
        return data_processing.random_batch(data, dataset_path,
                                            batch_size,
                                            img_size=self.IMG_SIZE,
                                            random_seed=random_seed,
                                            include_angles=self.ANGLES,
                                            include_speed=self.SPEED,
                                            nof_outputs=self.NOF_OUTPUTS)



class Bojarski_Model(Model):
    def get_model(self):
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu",
                         input_shape=self.INPUT_SHAPE))
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


class Bojarski_Model_Dropout(Model):
    def get_model(self):
        DROPOUT_RATE = 0.35
        print 'Using learning rate:', self.LEARNING_RATE

        adam = optimizers.adam(lr=self.LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Sequential()
        optimizer = optimizers.Adam(lr=0.0003)
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu",
                         input_shape=self.INPUT_SHAPE))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Flatten())
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Dense(10, activation="relu"))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Dense(self.NOF_OUTPUTS))
        model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mae'])
        return model


class Bojarski_Model2FC(Model):
    def get_model(self):
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu",
                         input_shape=self.INPUT_SHAPE))
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
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu",
                         input_shape=self.INPUT_SHAPE))
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
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu",
                         input_shape=self.INPUT_SHAPE))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(35, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.NOF_OUTPUTS))
        model.compile(optimizer="adam", loss="mse", metrics=['accuracy', 'mae'])
        return model


def get_model(name, learning_rate, include_speed=False):
    return {
        'Bojarski': Bojarski_Model(learning_rate, include_speed=include_speed),
        'Bojarski_Dropout': Bojarski_Model_Dropout(learning_rate, include_speed=include_speed),
        'Bojarski2FC': Bojarski_Model2FC(learning_rate, include_speed=include_speed),
        'Simplified_Bojarski': Simplified_Bojarski_Model(learning_rate, include_speed=include_speed),
        'Very_Simplified_Bojarski' : Very_Simplified_Bojarski_Model(learning_rate, include_speed=include_speed)
    }[name]
