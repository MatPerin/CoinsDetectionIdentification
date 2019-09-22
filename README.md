# Coins Detection and Identification
Identify and detect euro coins in a scene

# Objective of the program

The program should be able to detect the euro coins in a scene and recognize their face value.
The program should deal, therefore, with the problem of detection and, after that, identification.

# Step 1: Coins Detection

The first step of dealing with the problem is the detection of coins in the scene. In this step the
value of the coin is not important, we are only interest in identifying what is very likely to be a
coin. Coins are characterized by a circular shape, only one or two colour and a limited
variation in diameter. All these characteristics have been exploited for the correct detection.
First, the image is preprocessed: it is converted to graysacle, since we will be using the
Canny edge detector, and the noise is removed with a Gaussian Filter with kernel = [9; 9] and
theta = 1, this is because the method we will use to identify the circles, the generalized Hough
Transform, is really sensitive to noise. The image is also scaled proportionally to have an 800
pixel height, since the OpenCV implementation of HoughCircles() works best in smaller
images.

The most trivial choice for identifying circles in an image is the Hough Transform.
The problem is that we do not know in advance the radii of the coins and, for this
reason, many fake positives are found.

To solve this problem, the following steps are implemented:

1. A rough Hough Transform is applied, in this step we use an high Canny threshold to
get only the most connected circles but a low center threshold so we get most of the
circles.

2. The mean radius length across all found circles is computed.

3. A finer Hough Transform (higher center threshold) is applied, but this time we only
look for circles close to the mean radius. This step is not strictly needed, but it can
help speed up the following process by getting fewer circles.

4. The area inside each circle is considered. Each one in converted to HSV colour space
and the variance in the hue channel matrix is computed. Since coins are usually
constant in colour we can imply that the area with less variance (most constant colour)
is the most likely candidate to be a coin.

5. With this in mind, we have what is very likely to be the radius of a coin (by looking at
the size of the area with least variance in hue) and so we can apply an Hough Transform
with very low thresholds but limiting the radius to be very close to the found one (since
coins generally have similar diameters).

A class CoinsDetector has been built implementing this method. It has a single static
method: Detect() that takes in input the original image and gives in output the found circles
(vector of triples containing center coordinates and circle radius) and the rescaled image
used for detection.

# Step 2: Coins Identification

The identification of coins has been tackled with machine learning.

The dataset used for training and validation can be found at https://github.com/kaa/coins-dataset.
The training was dividied in batches of size 32 and went on for 200 epochs.
The tools used are Keras with a Tensorflow backend.

Despite the stucture of the Neural Network and the usage of data augmentation
during training, the results were not optimal due to the limitations in the
dataset: it is hard to train a multi class identifier when some classes have
less than 100 training samples. For this reason the maximum accuracy on
validation data that was obtained is about 70%, which is enough to identify
most of the coins correctly (albeit coins similar to each other, like cents, get
easily confused), but it can be vastly improved by providing more samples
to work with.

The model and its computed weights have then been loaded into OpenCV
using the dnn class. Each coin found (with the previously explained method)
is, at this point, rescaled to 150x150 pixels, converted to RGB (since OpenCV
uses BGR channel coding) and the colour values are rescaled to [0; 1] down
from [0; 255] (since the NN was trained on samples of this configuration).

The prediction is then calculated by calling the forward() method of the dnn
class. This gives back an array containing the confidence of the prediction
on each class. The class with the highest value is taken as the prediction of
the value of the coin.

The problem is that with only these steps also coins that are not euro
are recognised as such. To cope with this, the predictions which have a
certain amount of values that have a confidence above a certain threshold
are considered unsure and, therefore, discarded. This allows for the filtering
of some (but not all) fake positives, but the results could be better with a
more performant NN.

This implementation is, as can be seen, quite sensitive to the presence
of other coins in the image. On the other hand, using a machine learning

A class implementing the just stated method has been implemented: CoinsIdentificator.
It takes the path to the model weights and configuration files and an array with the map-
ping of the identifier ids (the dnn.forward() method gives back the confidence based on a
numeric id) into their respective strings. The prediction on each circle (found with Coins-
Detector.Detect()) can be retrieved by calling the Identify() method. It takes the rescaled
input image, the array containing the circles and the threshold in confidence and number
of values (difierent from the most likely prediction) that must have a likelihood above the
given threshold before rejecting the prediction. The method gives back a vector containing
the predictions (in string form) and another one containing the coins evaluated (coins near
the edge of the image are not evaluated since they give unsure results). Both vectors are in
the same order in which the circles are considered.

# Usage of the program

A test program has been implemented. It simply asks for the input image path. Then,
the coins are detected using the implemented class, and the resulting circles are drawn on
the image. After this, the coins are identified using the previously described class and the
predicted valued drawn on the image (or red circles are drawn if the prediction is unsure).
The resulting images are saved as "detection_filename.jpg" and "identification_filename.jpg".
