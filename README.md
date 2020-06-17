# Year-3-Project
University Of Surrey Electronic Engineering Year 3 Project On Facial Obfuscation and deepFakes
## REQUIREMENTS

* Python 3.7 or above
* dlib version 19.19.0 or above
* openCV version 4.2.0.32 or above
* numpy version 1.18.1 or above

You will also need a caffe model file for the SSD detector and a prototxt file
you will also need a a face landmark predictor which can be found in this repository

## HOW TO INSTALL DEPENDANCIES

In order to install these packages to Python you must use the following command:

"pip install dlib opencv-python numpy"

or

"pip3 install dlib opencv-python numpy"

if you have python 3 and python 2 install on the same system

## USAGE

"Python3 faceSwap.py --source example/face.jpg --destination example/destinationface.jpg --predictor predictorFILE.dat --model NNmodel.caffemodel --prototxt deploy.protoxt.txt --threshold 0.6 "

* --source : image file of source face
* --destination : Video file path
* --predictor : File path to face landmark predictor
* --model : File path to the Caffe model
* --prototxt : File path to prototxt for the ConvNet
* --threshold : Threshold for the face detection
