import cv2
import numpy as np
import os
import math

# Rotation matrix
def R(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    return np.ndarray(buffer=np.array([cos_theta, -sin_theta,
                                       sin_theta, cos_theta]), shape=(2, 2))


# MEASURED IRL

H = 0.23 # m - distance from camera to ground
D_0 = 0.44 # m - distance from bottom of car to first point on ground in camera view
Y_IRL = 0.57 # m - height of image frame at D_0 in real life
X_IRL = 0.5 #3 # m - width of image frame at D_0 in real life

IMAGE_HEIGHT = 66 # pixels
IMAGE_WIDTH = 200 # pixels

NOF_DOTS = 30
DRAW_EVERY_NTH_DOT = 1
FRAME_RATE = 15
CROPPING_PERCENTAGE = 0.65

F_THETA = 0.093


def calculate_axes(predictions, steering):
    h = []
    sum_d = D_0

    for prediction, steering in zip(predictions, steering):
        d_i = steering[1] / float(FRAME_RATE/DRAW_EVERY_NTH_DOT) # distance travelled in 1/15th second
        sum_d += d_i
        h_i = H*d_i / float(sum_d)
        h.append(h_i)

    r = []
    r_star = [np.array([0, h[0]])]
    r_1 = np.matmul(R(predictions[0]*F_THETA), r_star[0])
    r.append(np.array(r_1))
    p = np.zeros(shape= (len(h), 2))
    p[0] = np.array(r_1)

    for i in range(1, len(h)):
        r_i_star = r[i-1] * (h[i - 1] + h[i]) / float(h[i-1]) - r[i-1]
        r_star.append(r_i_star)

        r_i = np.matmul(R(predictions[i]*F_THETA), r_star[i])

        r.append(r_i)
        p_i = p[i - 1] + r[i]
        p[i] = np.array(p_i)

    return p[:,0], p[:,1]


def scale_vector(x_axis, y_axis, to_width, to_height):
    return x_axis * to_width / X_IRL, y_axis * to_height * CROPPING_PERCENTAGE / (Y_IRL)


def get_point_angle_data(validation_data, predictions, starting_point, dataset_path):
    X = starting_point
    indices = np.arange(X, X + NOF_DOTS*DRAW_EVERY_NTH_DOT, DRAW_EVERY_NTH_DOT)

    preview_image = cv2.imread(os.path.join(dataset_path, validation_data['image_names'][indices[0]]))
    preview_image = cv2.resize(preview_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    wanted_predictions = predictions[indices]

    x_axis, y_axis = calculate_axes(wanted_predictions, validation_data['steers'][indices])
    true_x_axis, true_y_axis = calculate_axes(validation_data['steers'][:,0][indices], validation_data['steers'][indices])

    x_axis, y_axis = scale_vector(x_axis, y_axis, IMAGE_WIDTH, IMAGE_HEIGHT)
    x_axis = x_axis + IMAGE_WIDTH / 2
    y_axis = np.negative(y_axis) + IMAGE_HEIGHT

    true_x_axis, true_y_axis = scale_vector(true_x_axis, true_y_axis, IMAGE_WIDTH, IMAGE_HEIGHT)
    true_x_axis = true_x_axis + IMAGE_WIDTH / 2
    true_y_axis = np.negative(true_y_axis) + IMAGE_HEIGHT

    return x_axis, y_axis, true_x_axis, true_y_axis, preview_image
