import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import random
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from PIL import Image


# Initialize main parameters
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (2000/4, 256/4, 1)
data = []
labels = []

""" LOAD THE IMAGES """
print('LOADING DATA...')
directory = '/Users/greysonbrothers/Desktop/ /- python/- data science/PROJECTS/- Neural Networks/TOILET PROJECT/training_images/'
for folder in os.listdir(directory):
    if folder in ['.DS_Store']:
        continue    
    # loop over the input images
    for imagePath in os.listdir(directory + folder + '/'):
        if imagePath == '.DS_Store':
            continue
        # load the image, pre-process it, and store it in the data list
        image = Image.open(directory + folder + '/' + imagePath)
        image = image.resize((int(IMAGE_DIMS[1]), int(IMAGE_DIMS[0])))
        image = np.asarray(image) 
        data.append(image)
        labels.append(folder) 
    print('.')
print('LOADING COMPLETE')


""" ASSIGN DATA TO TESTING SETS """
data = np.array(data)/255
labels = np.array(labels)
# 60 - 20 - 20 split of the data
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

""" ENCODE LABELS """
encoder = LabelEncoder()
y_train = np_utils.to_categorical(encoder.fit_transform(Y_train))
y_val = np_utils.to_categorical(encoder.fit_transform(Y_val))
y_test = np_utils.to_categorical(encoder.fit_transform(Y_test))
num_classes = len(y_test[0])


""" CONSTRUCT THE MODEL """
print('\nBUILDING MODEL')
model = Sequential()
input_shape = IMAGE_DIMS
pool = 2

#1 CONV => RELU => POOL
model.add(Conv1D(32, 3, padding='same', activation='relu', input_shape=(int(IMAGE_DIMS[0]), 64)))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling1D(pool_size=pool))
          
#2 (CONV => RELU) * 2 => POOL            
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling1D(pool_size=pool))
downsize = pow(pool, 2)


#3 (CONV => RELU) * 2 => POOL            
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling1D(pool_size=pool))
downsize = pow(pool, 3)
model.add(Dropout(0.1))

#4 (CONV => RELU) * 2 => POOL            
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling1D(pool_size=pool))
downsize = pow(pool, 4)
model.add(Dropout(0.1))

#5 (CONV => RELU) * 2 => POOL            
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling1D(pool_size=pool))
downsize = pow(pool, 5)
model.add(Dropout(0.1))

#6 (CONV => RELU) * 2 => POOL            
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling1D(pool_size=pool))
downsize = pow(pool, 6)
model.add(Dropout(0.15))

#7 (CONV => RELU) * 2 => POOL            
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling1D(pool_size=pool))
model.add(Dropout(0.1))
model.add(Flatten())

# FULLY CONNECTED LAYER
model.add(Dense(64*64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# FINAL LAYER SOFTMAX
model.add(Dense(num_classes))
model.add(Activation('softmax'))


""" TRAIN THE MODEL """
print("TRAINING MODEL...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=BS, epochs=EPOCHS, validation_data=(X_val, y_val))

""" PRINT ACCURACY """     
result = model.predict(X_test)
correct = [(np.argmax(y_test[i])== np.argmax(result[i])) for i in range(len(y_test))]
acc = round(100*sum(correct)/len(correct), 2)
print("Accuracy: ", round(acc, 2), "%", sep='')

# Plot Confusion Matrix
y_pred = [np.argmax(i) for i in result]
y_pred = pd.DataFrame(encoder.inverse_transform(y_pred))
conf_mat = confusion_matrix(Y_test, y_pred, labels=encoder.classes_)
conf_mat = np.array([np.around(i/sum(i),2) for i in conf_mat])  # normalize entries
fig, ax = plt.subplots(figsize=(9,8))
ax = sns.heatmap(conf_mat, cmap='Blues', annot=True, xticklabels=encoder.classes_, yticklabels=encoder.classes_)
ax.tick_params(labelsize=6)
plt.title('2 Accuracy: ' + str(acc))
plt.xlabel('Predictions')
plt.ylabel('True Values')
plt.tight_layout()

# Print Recall & Precision Metrics
recall = np.mean([row[i]/sum(row) for i,row in enumerate(conf_mat)])
precision = np.mean([row[i]/sum(row) for i,row in enumerate(conf_mat.T)])
print('Avg Recall: ', recall)
print('Avg Precision: ', precision)

# Plot training & validation accuracy values
fig1 = plt.figure(figsize=(6,8))
ax1 = fig1.add_subplot(2,1,1)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
plt.title('2 Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
ax2 = fig1.add_subplot(2,1,2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
plt.title('2 Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.tight_layout()
