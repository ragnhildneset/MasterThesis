#!/usr/bin/env python

import os
import errno
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import rospy
from std_msgs.msg import Int8, UInt16
from sensor_msgs.msg import Image
import pandas as pd


# Global variables
rate = 10
frame = 0
current_steering = 0
current_throttle = 0
writeHeader = True
init = False


name = raw_input('Dataset Name: ')
path = os.path.expanduser("~") + "/" + name + '/' + 'Original/'
csv_name = 'labels.csv'
raw_input('Saving to: ' + path + csv_name + '. Press Enter to continue')


def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise


make_sure_path_exists(path)

bridge = CvBridge()
buffer_size = 1
image_buffer = []
label_buffer = []


def save_data(image, labels):
	label_buffer.append(labels)
	image_buffer.append(image)
	if len(label_buffer) >= buffer_size:
		csvString = ''.join([','.join([str(x['steering']),
									   str(x['throttle']),
									   x['image']]) + '\n' for x in label_buffer])
		write_csv(csvString)

		for i in range(0, len(image_buffer)):
			imgName = path + label_buffer[i]['image']
			cv2.imwrite(imgName, image_buffer[i])
		label_buffer[:] = []
		image_buffer[:] = []


def write_csv_header():
	with open(path + csv_name, 'w+') as file:
		file.write('steering,throttle,image\n')


def write_csv(text):
	with open(path + csv_name, 'a') as file:
		file.write(text)


def steering_callback(data_loc):
	global current_steering
	current_steering = data_loc.data


def throttle_callback(data_loc):
	global current_throttle
	current_throttle = data_loc.data


def image_callback(data_loc):
	global frame

	frame += 1
	labels = {}
	labels["steering"] = current_steering
	labels["throttle"] = current_throttle
	labels["image"] = "image" + str(frame) + ".bmp"
	save_data(bridge.imgmsg_to_cv2(data_loc, "bgr8"), labels)
	if frame % 100 == 0:
		print('Frames:', frame)



def countdown(message, seconds):
	while seconds > 0:
		print(message, 'in', seconds, 'seconds')
		time.sleep(1)
		seconds -= 1


def run_collector():
	frame = 0


	write_csv_header()


	rospy.Subscriber("/arduino/steering_current", Int8, steering_callback)
	rospy.Subscriber("/arduino/throttle_current", UInt16, throttle_callback)
	rospy.Subscriber("/zed/left/image_rect_color", Image, image_callback)
	rospy.init_node('collect_training_data', anonymous=False)

	raw_input("Press Enter at any time to cancel the collecting...\n")

if __name__ == "__main__":
	countdown("Starting capture", 10)
	run_collector()
