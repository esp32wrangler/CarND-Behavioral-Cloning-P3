import tensorflow
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import ceil

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#different data set experiments
csvfile1 = open ("recordings/one/driving_log.csv", "r")
csvfile2 = open ("recordings/two/driving_log.csv", "r")
csvfile3 = open ("recordings/three/driving_log.csv", "r")
csvfile4 = open ("recordings/four/driving_log.csv", "r")
csvfile5 = open ("recordings/five/driving_log.csv", "r")
csvfile6 = open ("recordings/six/driving_log.csv", "r")
csvfile7 = open ("recordings/seven/driving_log.csv", "r")
csvfile8 = open ("recordings/eight/driving_log.csv", "r")
csvfile9 = open ("recordings/nine/driving_log.csv", "r")
csvfile10 = open ("recordings/ten/driving_log.csv", "r")
csvfile11 = open ("recordings/eleven/driving_log.csv", "r")
csvfile12 = open ("recordings/twelve/driving_log.csv", "r")

sidecorrection = 0.18

X_train = []
y_train = []
#final dataset combining the goldilocks combination of experiments above
lines = csvfile1.readlines() + csvfile2.readlines() + csvfile3.readlines() + csvfile4.readlines()  + \
        csvfile5.readlines()  + csvfile6.readlines()  + csvfile7.readlines() + csvfile12.readlines()

#split dataset to validation and training samples
train_samples, valid_samples = train_test_split(lines, test_size=0.2)

#read and preprocess images in batches
#6 images generated: center original, center flipped, right camera view original and flipped, left camera and flipped
# off-center images associated with a +-0.18 degree steering angle to simulate returning to centerline
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines=lines[offset:offset+batch_size]

            X_train=[]
            y_train=[]
            for line in batch_lines:
                cimgname, limgname, rimgname, steer, throttle, brake, speed = line.split(",")
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
            yield shuffle(X_train, y_train)

batch_size = 32

train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(valid_samples, batch_size=batch_size)

model = keras.models.Sequential ()
# normalize pixel values to -1..1
model.add(keras.layers.Lambda(lambda x: (x-128)/128, input_shape=(160,320,3)))
# crop top of image (typically it shows trees, sky and other irrelevant details, but on the second track this
#   also means sacrificing some of the road ahead when there is a steep upward slope)
model.add(keras.layers.Cropping2D(cropping=( (45, 25), (0, 0))))

# Modified LeNet architecture with more channels and an additional convolutional layer
model.add(keras.layers.Conv2D(12, (5,5), activation="relu"   ))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(24, (5,5), activation="relu" ))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(36, (5,5), activation="relu" ))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense (120, activation="relu" ))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense (84, activation="relu" ))
model.add(keras.layers.Dense (1))

model.compile(optimizer='adam', loss='mse')

history_object = model.fit_generator(train_generator, validation_data=valid_generator,
                                     steps_per_epoch=ceil(len(train_samples)/batch_size),
                                     validation_steps=(len(valid_samples)/batch_size),
                                     epochs=3, shuffle=True, verbose=1  )

# save the model so the drive.py module can load it
model.save('model.h5')

#plot an epoch-by-epoch overview of the mean sq loss for training and validation data
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.ylabel ('mean sq loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()
