# Toilet-Flush-Convolutional-Neural-Net

Constructed a convolutional neural network to classify audio files as being one of 
the classes [toilet, urinal, sink, unknown].  The model was built using a VGG16 architecture and the dataset was 
augmented with 3 levels of noise reduction per clip and horizontal translation of the 
clip accross the 30 second input window used as the feature vector.  Currently working 
on collecting & augmenting data to build the final model.  The purpose of this 
project is to automate the detection and counting of toilet/urinal/sink uses in a given 
time period from an audio device so that the amount of water used by a given bathroom 
over a specified time period can be calculated. This project was motivated by work I'm 
doing for Bruin Home Solutions, a sustainability club I run at UCLA.  

Dataset: https://urbansounddataset.weebly.com/download-urbansound.html
