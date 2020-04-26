# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center_2020_04_26_08_42_17_218.jpg "Centerline"
[image2]: ./writeup_images/center_2020_04_26_08_42_17_218f.jpg "Flipped"
[image3]: ./writeup_images/left_2020_04_26_08_42_17_218.jpg "Left"
[image4]: ./writeup_images/left_2020_04_26_08_42_17_218f.jpg "Left Flipped"
[image5]: ./writeup_images/right_2020_04_26_08_42_17_218.jpg "Right"
[image6]: ./writeup_images/right_2020_04_26_08_42_17_218f.jpg "Right Flipped"
[image7]: ./examples/placeholder_small.png "Recovery Image"
[image8]: ./examples/placeholder_small.png "Recovery Image"
[image9]: ./examples/placeholder_small.png "Normal Image"
[image10]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 video showing how the model drove around on Track 2

These can be found in my GitHub [repository]( https://github.com/esp32wrangler/CarND-Behavioral-Cloning-P3 )

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. 
The file shows the pipeline I used for training and validating the model,
and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network that is based on the LeNet architecture, but expanded for a bit better performance (see lines 81-93)
The model includes RELU layers to introduce non-linearity, and the input pixel values are normalized to the [-1..1] range using a Keras lambda layer (code line 77). 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (see table above). 

The model was trained and validated on different data sets to reduce the chance of the model 
overfitting. I recorded two drives in the correct direction and one drive in the opposite direction
to add variety. For each car position, 6 images are introduced into the dataset, with their corresponding steering angle as the desired output:
* The center camera's image in both as-is, and a horizontally flipped
* The left camera's image normal and flipped, with a +-0.18 steering angle correction to account for the difference in viewpoint 
* The right camera's image normal and flipped, with a -+0.18 steering angle correction to account for the difference in viewpoint 
 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, normal driving with corner cutting, driving in the opposite direction,
and specific recovery scenarios in tricky situations (starting at a car location and attitude that is clearly off-course
but still recoverable, and recovering from it).  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to record several different drives through the track, and then augment 
this dataset as necessary based on the model performance.

I started out with a traditional LeNet architecture with two convolutional and three fully connected layers. This is 
a general architecture that has performed very well in number and road sign recognition tasks, so I expected
it would have success with detecting road surfaces and road markings as well, especially when scaled up to the 
rather large size of the input image.
This architecture allowed me to build a model that was capable of driving the first track, but for 
the second course I was seeing signs of serious underfitting, so I decided to beef up the number of channels and introduce
an additional convolutional layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I visualized how the mean square error was changing for the training and validation data set over the epochs, and looked for
signs of overfitting (the validation performance getting worse while training performance was improving) and underfitting
(failure to improve both the test and validation performance). To reduce overfitting I used dropout layers and
as many different image variations as possible, and to improve the underfitting I extended the model with layers and channels. 

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track. In particular on the first track there was 
a section where a dirt road branches off from the paved road, and the model was not able to detect the 
edge of the road between the paved and dirt sections. To improve the driving behavior in these cases, I 
decided to introduce additional data points focusing on this particular section by driving slowly and recovering from the 
incorrect heading toward the dirt turn-off (I drove slowly to record as many frames as possible with a single drive).

On the second track the location causing the most problems was the right turn at the hilltop (2:30 in run1.mp4). 
At this point the road past the mountain top, in the valley looks like it is connected to the road surface we are driving on, and the 
model consistently mistook the valley road to be a straight continuation of "our" road. When I added a few more drivethroughs
in this area, I started getting other errors showing unbalance in the training data. Finally I had to carefully record just the recovery
scenario to get a balanced training result that could handle all the weird situations on the 
track (blind curving drops, steep uphills that went into the top clipping area of the image, sharp turns before steep dropoffs, etc.)

At the end of the process, the vehicle is able to drive autonomously around both of the provided tracks without leaving the road.

#### 2. Final Model Architecture

Here is the design of the final network (see also lines 81-93). It is probably a bit of an overkill, but my
GPU was really suffering on this model, so I didn't have the patience to fine-tune it and cut it down to 
the minimum viable size.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image, cropped to 250x160x3		| 
| C1 Convolution 5x5   	| 1x1 stride, 12 output channels                 |
| Max pooling, RELU    	| 2x2 stride                       				|
| C2 Convolution 5x5	| 1x1 stride, 24 output channels                 	|
| Max pooling, RELU		| 2x2 stride									|
| C3 Convolution 5x5	| 1x1 stride, 36 output channels                 	|
| Max pooling, RELU		| 2x2 stride									|
| Flattening		    | flatten to 1D             					|
| Dropout   	      	| 0.3 drop probability             				|
| F1 Fully connected	| outputs 120  									|
| RELU					|												|
| Dropout   	      	| 0.3 drop probability             				|
| F2 Fully connected	| outputs 84  									|
| RELU					|												|
| Dropout   	      	| 0.3 drop probability             				|
| F3 Fully connected	| outputs final steering angle         			|
| 						|						


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track. One being slow and very
 careful to follow the centerline, and another one driving at full speed cutting the corners as appropriate.

![alt text][image1]

I then recorded the track in the opposite direction, with some zig-zag to add correction behavior to the model.

As mentioned above, I augmented the dataset with flipped images from left, center and right cameras to add variety and simulate
off-center driving.

Here are the six images for the position above: 

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

After the collection process, I had 5442 locations for the first course and 11878 locations for the second track.
Multiplying this with the 6 views for each location gives just over 50000 training images. I ended up adding
3500 more locations to balance the dataset and successfully complete autonomously driving the second track.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 3 as evidenced by no further improvement in test and validation mean squared error in subsequente epochs.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
