from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D


def model():
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
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model
