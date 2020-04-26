import tensorflow
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

csvfile1 = open ("recordings/one/driving_log.csv", "r")
csvfile2 = open ("recordings/two/driving_log.csv", "r")
csvfile3 = open ("recordings/three/driving_log.csv", "r")
csvfile4 = open ("recordings/four/driving_log.csv", "r")
csvfile5 = open ("recordings/five/driving_log.csv", "r")
csvfile6 = open ("recordings/six/driving_log.csv", "r")
csvfile7 = open ("recordings/seven/driving_log.csv", "r")

sidecorrection = 0.18

X_train = []
y_train = []
lines = csvfile1.readlines() + csvfile2.readlines() + csvfile3.readlines() + csvfile4.readlines()  + csvfile5.readlines()  + csvfile6.readlines()  + csvfile7.readlines()
for line in lines:
    cimgname, limgname, rimgname, steer, throttle,brake,speed = line.split(",")
    cimg = mpimg.imread(cimgname)
    crop_cimg = cimg[:, :]
    limg = mpimg.imread(limgname)
    crop_limg = limg[:, :]
    rimg = mpimg.imread(rimgname)
    crop_rimg = rimg[:, :]
    X_train.append(crop_cimg)
    y_train.append(float(steer))
    X_train.append(crop_limg)
    y_train.append(float(steer) + sidecorrection)
    X_train.append(crop_rimg)
    y_train.append(float(steer) - sidecorrection)
    X_train.append(cv2.flip(crop_cimg, 1))
    y_train.append(-float(steer))
    X_train.append(cv2.flip(crop_limg, 1))
    y_train.append(-float(steer) - sidecorrection)
    X_train.append(cv2.flip(crop_rimg, 1))
    y_train.append(-float(steer) + sidecorrection)

X_train = np.array(X_train)
y_train = np.array(y_train)

model = keras.models.Sequential ()
model.add(keras.layers.Lambda(lambda x: (x-128)/128, input_shape=(160,320,3)))
model.add(keras.layers.Cropping2D(cropping=( (65, 25), (0, 0))))

#model.add(keras.layers.Lambda(lambda x: (x-128)/128, input_shape=(160,320,3)))
model.add(keras.layers.Conv2D(6, (5,5), activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(16, (5,5), activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense (120, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense (84, activation="relu"))
model.add(keras.layers.Dense (1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=3, validation_split=0.2, shuffle=True  )

model.save('model.h5')