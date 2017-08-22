#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/rbg2gray.png "Grayscaling"
[image3]: ./examples/augmented.png "Augmented Image"
[image4]: ./examples/blurred.png "Blurred Image"
[image5]: ./examples/preprocessing.png "Visualization Preprocessing"
[image6]: ./new_image/11_100_1607_small.jpg "Traffic Sign 1"
[image7]: ./new_image/12_Arterial_small.jpg "Traffic Sign 2"
[image8]: ./new_image/14_Stop_sign_small.jpg "Traffic Sign 3"
[image9]: ./new_image/17_Do-Not-Enter_small.jpg "Traffic Sign 4"
[image10]: ./new_image/36_Untitled.jpg "Traffic Sign 5"
[image11]: ./examples/stop_top5.png "Traffic Sign 1 prediction"
[image12]: ./examples/straight_top5.png "Traffic Sign 2 prediction"
[image13]: ./examples/priority_top5.png "Traffic Sign 3 prediction"
[image14]: ./examples/NoEntry_top5.png "Traffic Sign 4 prediction"
[image15]: ./examples/RWNI_top5.png "Traffic Sign 5 prediction"
[image16]: ./examples/act1.png "act1"
[image17]: ./examples/act2.png "act2"
## Rubric Points

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Visualize and count the number of each class of the dataset.

Here is an exploratory visualization of the original data set. It is a bar chart showing how the data distributed.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data.

As a first step, I decided to convert the images to grayscale because it not only reduces the input volume (32x32x3 -> 32x32x1), but also keep all the information for artificial identification. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then I utilize [ImageDataGenerator](https://keras.io/preprocessing/image/) to generate fake data based on the original data set. The ImageDataGenerator can generate fake data by randomly rotating (< 17 degrees), randomly shifting in width and height (< 20%) and randomly zooming (< 15%). Also, to deal with the unbalanced training data set, I generated 1000 more augmented images for the classes which have less than 1000 samples.

Here is an example of a traffic sign image and an augmented image.

![alt text][image3]

Furthermore, I implement Gaussian Blur to generate more blurred images for both original and augmented images.

Here is an example of a traffic sign image and an blurred image.

![alt text][image4]

As a last step, I normalized the image data because the means and standard deviations of features will have difference without normalization, the learning progress of CNN will correct the difference. If the learning rate is defined as a scalar, the learning progress may cause overcompensating in one weight dimension as well as undercompensating in another.

I decided to generate additional data for minority class because the unbalance date set will cause the model tend to classify most results to the majority class based on the metrics of prediction accuracy only.

Finally, the volume of the training data set is 172,978. Here is an exploratory visualization of the preprocesing data set. It is a bar chart showing how the data distributed.

![alt text][image5]

####2. Describe what your final model architecture.

I developed two models, and both of them achieve the test accuracy to be 0.969. The model 0 consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					| act1											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Dropout				| keep probability = 0.75						|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64	|
| RELU					| act2											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Dropout				| keep probability = 0.75						|
| Fully connected 0		| nodes = 5x5x64 = 1600							|
| Fully connected 1		| nodes = 800									|
| Fully connected 2		| nodes = 200									|
| Softmax				| logits = 43  									|

The model 1 consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					| act1											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Dropout				| keep probability = 0.75						|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64	|
| RELU					| act2											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Dropout				| keep probability = 0.75						|
| Fully connected 0		| nodes = 5x5x64 = 1600							|
| Fully connected 1		| nodes = 500									|
| Fully connected 2		| nodes = 150									|
| Softmax				| logits = 43  									|

####3. Describe how you trained your model. 

To train the model, I used LeNet-5 as a start, which get a validation set accuracy of about 0.90. During the turning process, I keep the batch size to 128 and optimizer to AdamOptimizer with learning rate 0.001. The major process of hyperparameters turning includes following steps: 

* I increase the convolution filter size from 3x3 to 5x5, which doesn't improve the performance. But I keep the size to be 5x5, since I think the Traffic Sign image should have larger filter than the original LeNet-5 model for digits.
* By adding dropout, the difference between the validation accuracy and test accuracy is reduced, which indicates the overfitting problem is improved by dropout.
* Also, L2 regularizer is introduced to constrain the complexity of the weights, which reduce the overfitting.
* After turning the depth of each convolution layer and the number of nodes for each fully connected layers, the increase of the depth to 32 in convolution layer 1 and the depth to 64 in convolution layer 2 make a distinguished improvement on accuracy. 
* Finally, the number of epochs is defined as 30. The best validation accuracy of the trained model during the 30 epochs is saved to be the final model.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.982 
* test set accuracy of 0.969

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The second image might be difficult to classify because the right bottom part has a building connecting with the sign. And the fourth image might be difficult to classify since the complex background. The rest images should be easy to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

| Image			        					|     Prediction	     	  					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Stop Sign      							| Stop sign   									| 
| Go straight or right						| Go straight or right							|
| Priority									| Priority										|
| No entry	      							| No entry						 				|
| Right-of-way at the next intersection		| Right-of-way at the next intersection			|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.9%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model is sure that this is a stop sign (probability of 0.2, which is 2.8 times of the top two probability of 0.07), and the image does contain a stop sign.  Since the top two to five predictions contain circle sign, the result is acceptable. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .192         			| Stop sign   									| 
| .073     				| Speed limit (60km/h)							|
| .040					| Turn right ahead								|
| .035	      			| Speed limit (30km/h)			 				|
| .029				    | End of speed limit (80km/h)					|

![alt text][image11]



For the second image, the model is sure that this is a Go straight or right sign (probability of 0.25, which is 2.5 times of the top two probability of 0.1), and the image does contain a Go straight or right sign. Since the top two to five predictions contain arrow pointing up or right, the result is acceptable. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .258         			| Go straight or right							| 
| .100     				| Ahead only									|
| .065					| Turn right ahead								|
| .045	      			| Keep right					 				|
| .020				    | Dangerous curve to the right					|

![alt text][image12]

For the third image the model is sure that this is a Priority road sign (probability of 0.30, which is 2 times of the top two probability of 0.15), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .298         			| Priority road									| 
| .152     				| Roundabout mandatory							|
| .069					| Speed limit (50km/h)							|
| .054	      			| Speed limit (100km/h)				 			|
| .044				    | No entry		      							|

![alt text][image13]

For the fourth image the model is pretty sure that this is a No entry sign (probability of 0.50, which is almost 3 times of the top two probability of 0.18), and the image does contain a No entry sign. The top five soft max probabilities were 

| Probability         	|     Prediction	   		     						| 
|:---------------------:|:-----------------------------------------------------:| 
| .505         			| No entry   											| 
| .184     				| Speed limit (100km/h)									|
| .121					| Keep right											|
| .114	      			| End of no passing by vehicles over 3.5 metric tons	|
| .065				    | Priority road     									|

![alt text][image14]

For the fifth image the model is pretty sure that this is a Right-of-way at the next intersection sign (probability of 0.52, which is almost 2 times of the top two probability of 0.25), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .518         			| Right-of-way at the next intersection			| 
| .252     				| Beware of ice/snow							|
| .147					| Double curve									|
| .110	      			| Dangerous curve to the right	 				|
| .073				    | General caution      							|

![alt text][image15]

### (Optional) Visualizing the Neural Network 
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Here is the output of convolution layer 1 for the Right-of-way at the next intersection sign. The feature 2, feature 14, and feature 17 indicates it contain a triangle shape, which may have high weights to predict to be a Right-of-way at the next intersection sign.

![alt text][image16]

Here is the output of convolution layer 2 for the Right-of-way at the next intersection sign. However, I cannot figure any useful information for prediction.

![alt text][image17]