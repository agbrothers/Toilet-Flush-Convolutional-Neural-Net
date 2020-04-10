# Toilet-Flush-Convolutional-Neural-Net

Constructing a convolutional neural network to classify audio files as being one of 
the classes [toilet, urinal, sink, unknown].  First a test model was built and trained 
on an environmental audio dataset called UrbanSounds with 9 sound classes, and a 
classification accuracy of 96.84% was achieved.  The model was built using a VGG 
architecture and the dataset was augmented with 3 levels of noise reduction 
per clip and horizontal translation of the clip accross the 30 second input window used 
as the feature vector.  Currently working on collecting &amp; augmenting data to build 
the final model.  The purpose of this project is to automate the detection and counting 
of toilet/urinal/sink uses in a given time period from gathered audio such that the 
amount of water used in a given bathroom per time period can be calculated.  This work 
was motivated by Bruin Home Solutions, a sustainability club I run at UCLA.  

Dataset: https://urbansounddataset.weebly.com/download-urbansound.html
