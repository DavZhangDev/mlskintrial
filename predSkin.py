# Import required libraries
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Input, Activation, Dropout, Flatten, Dense
from IPython.display import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

IMAGE_WIDTH, IMAGE_HEIGHT = 200, 200
EPOCHS = 30
BATCH_SIZE = 32
num_classes = 2

savedmodel = "/savemodel/skinpredv5.hdf5"
predimgpath = "BenignSkin.jpg"
fpath = "img/"

'''
conda create -n tf tensorflow
conda activate tf
'''

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return str(int(n * multiplier) / multiplier)

def defineSkinModel():
   model = Sequential()
   model.add(Conv2D(128, (3, 3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01), name = 'Conv_01'))
   model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_01'))
   model.add(Conv2D(96, (3, 3), padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01), name = 'Conv_02'))
   model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_02'))
   model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01), name = 'Conv_03'))
   model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_03'))
   model.add(Flatten(name = 'Flatten'))
   model.add(Dense(16, activation = 'relu', name = 'Dense_01'))
   model.add(Dense(2, activation='softmax', name = 'Output')) # 2 because of number of classes
   return model

def importSavedSkinModel():
    model = defineSkinModel()
    opt = Adam(lr = 0.0001)
    model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = opt, metrics = ['acc'])
    checkpointer = ModelCheckpoint(filepath="/savemodel/skinpred.hdf5", verbose=1, save_best_only=True)
    return model

def loadSkinModel():
    model = importSavedSkinModel()
    return model

def makePredictionGen():
    predlist = [fpath + predimgpath]
    prediction_df = pd.DataFrame({'filename': predlist, 'class': ["blank"]})
    prediction_data_generator = ImageDataGenerator(rescale=1./255)
    prediction_generator = prediction_data_generator.flow_from_dataframe(prediction_df, target_size = (IMAGE_WIDTH, IMAGE_HEIGHT), batch_size = 1, shuffle = False, class_mode = "categorical")
    return prediction_generator

def printPreds(predResult):
    # Converts prediction results into console log
    print('\nProbabilities:')
    if predResult[0] == 'b':
        print("This area of skin appears to be benign, but you should still check it out at the doctor's if you aren't sure!")
        print("Your skin abnormality is", predResult[1] + "% benign, " + predResult[2] + "% malignant.")
    else:
        print("That's no ordinary abnormality, you should definitely check it out at the doctor's!")
        print(predResult[1] + "% benign, " + predResult[2] + "% malignant.")

def returnPreds(datagen, pred_model):
    images, labels = datagen.next()
    img = images[0]
    preds = importSavedSkinModel().predict(np.expand_dims(img, axis = 0))
    ps = preds[0] # The raw prediction scores of the model
    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)
    classes = ['benign', 'malignant']
    if ps[0] > 0.5:
        return ["b",truncate(ps[0]*100, 1),truncate(ps[1]*100, 1)]
    else:
        return ["m",truncate(ps[0]*100, 1),truncate(ps[1]*100, 1)]

printPreds(returnPreds(makePredictionGen(), loadSkinModel()))

print("\nPREDICTION OF", predimgpath, "COMPLETE")
