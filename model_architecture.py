import matplotlib
matplotlib.use("Agg") # set the matplotlib backend so figures can be saved in the background
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, clone_model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from PIL import Image




def load_model(name):
    model_path = '.../models/'+name
    with open(model_path + '.json', 'r') as file:
        model = model_from_json(file.read())
    model.load_weights(model_path + '.h5')
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return(model)


def copy_model(model, dim):
    model_copy= clone_model(model)
    model_copy.build((None, dim)) # replace 10 with number of variables in input layer
    model_copy.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    model_copy.set_weights(model.get_weights())
    return model_copy


def save_model(model, name):
    path = '.../models/' + name
    model_json = model.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path+".h5")
    

def build_model(num_classes, input_dim=(500, 64, 1), epochs=100, bs=32, lr=1e-3, init='normal'):
    #1 CONV => CONV => POOL
    model = Sequential()
    model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_dim))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=2))
              
    #2 CONV => CONV => POOL         
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=2))
    
    #3 CONV => CONV => CONV => POOL           
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    
    #4 CONV => CONV => CONV => POOL             
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1)) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    
    #5 CONV => CONV => CONV => POOL               
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv1D(512, 3, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))    
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    
    #6 DENSE-1   
    model.add(Dense(14*4*512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    #7 DENSE-2 
    model.add(Dense(64*64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))   

    # DENSE-3 => OUTPUT
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # COMPILE MODEL
    opt = Adam(lr=lr, decay=lr/epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

    
def load_dataset(directory):
    print('LOADING DATA...')
    data,labels = [],[]
    for folder in os.listdir(directory):
        if folder != '.DS_Store':
            for image_path in os.listdir(directory + folder + '/'):
                if image_path != '.DS_Store':
                    image = Image.open(directory + folder + '/' + image_path)
                    image = np.asarray(image) 
                    data.append(image)
                    labels.append(folder) 
            print('.')
    print('LOADING COMPLETE')
    return np.array(data)/255, np.array(labels)


def plot_conf_mat(result, encoder):
    y_pred = [np.argmax(i) for i in result]
    y_pred = pd.DataFrame(encoder.inverse_transform(y_pred))
    conf_mat = confusion_matrix(Y_test, y_pred, labels=encoder.classes_)
    conf_mat = np.array([np.around(i/sum(i),2) for i in conf_mat])  # normalize entries
    fig, ax = plt.subplots(figsize=(9,8))
    ax = sns.heatmap(conf_mat, cmap='Blues', annot=True, xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    ax.tick_params(labelsize=6)
    plt.title('Confusion Matrix')
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.tight_layout()
    return conf_mat
    
    
def plot_loss(history):
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
      
    
    
if __name__ == "__main__":
    
    """ LOAD DATA """
    directory = '.../training_images/'
    X,y = load_dataset(directory)
    # 60 - 20 - 20 split of the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)
    
    """ ENCODE LABELS """
    encoder = LabelEncoder()
    y_train = np_utils.to_categorical(encoder.fit_transform(Y_train))
    y_val = np_utils.to_categorical(encoder.fit_transform(Y_val))
    y_test = np_utils.to_categorical(encoder.fit_transform(Y_test))
    num_classes = len(y_test[0])
    
    """ BUILD + TRAIN """
    model = build_model(num_classes)
    history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))
    
    """ MODEL EVALUATION """  
    plot_loss(history)
    result = model.predict(X_test)
    accuracy = model.evaluate(X_test,y_test)
    conf_mat = plot_conf_mat(result, encoder)
    recall = round(np.mean([row[i]/sum(row) for i,row in enumerate(conf_mat)])*100, 2)
    precision = round(np.mean([row[i]/sum(row) for i,row in enumerate(conf_mat.T)])*100, 2)
   
    print("Accuracy: ", round(accuracy, 2), "%", sep='')
    print('Avg Recall: ', recall, '%',sep='')
    print('Avg Precision: ', precision, '%', sep='')
        
    """ SAVE MODEL """
    model_name = "acc_"+str(accuracy)+"_recall_"+str(recall)
    save_model(model, model_name)


