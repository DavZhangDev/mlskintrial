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
from fx import *
import keras

IMAGE_WIDTH, IMAGE_HEIGHT = 200, 200
EPOCHS = 15
BATCH_SIZE = 32
num_classes = 2
'''
conda create -n tf tensorflow
conda activate tf
'''

def create_model():
   model = Sequential()
   model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01), name = 'Conv_01'))
   model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_01'))
   model.add(Conv2D(72, (3, 3), padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01), name = 'Conv_02'))
   model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool_02'))
   model.add(Flatten(name = 'Flatten'))
   model.add(Dense(16, activation = 'relu', name = 'Dense_01'))
   model.add(Dense(2, activation='softmax', name = 'Output')) # 2 because of number of classes
   return model

def lm():
    model = create_model()
    opt = Adam(lr = 0.0001)
    model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = opt, metrics = ['acc'])
    checkpointer = ModelCheckpoint(filepath="/savemodel/skinpred.hdf5", verbose=1, save_best_only=True)
    return model

def loadtrainedmodel(weights_path):
    model = create_model()
    model.load_weights(weights_path)

model = lm()

fpath = ""
predlist = [fpath + "Melanoma.jpg", fpath + "Mole.jpg", fpath + "Skin.jpg", fpath + "Cancer.jpg"]
prediction_df = pd.DataFrame({'filename': predlist, 'class': ["malignant", "benign", "benign", "malignant"]})
prediction_data_generator = ImageDataGenerator(rescale=1./255)
prediction_generator = prediction_data_generator.flow_from_dataframe(prediction_df,
                                             target_size = (IMAGE_WIDTH, IMAGE_HEIGHT),
                                             batch_size = 1,
                                             shuffle = False,
                                             class_mode = "categorical")

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return str(int(n * multiplier) / multiplier)

def pred_classify(datagen, pred_model):
    # Get first batch
    images, labels = datagen.next()

    img = images[0]
    print(img.ndim)
    print(img)

    preds = model.predict(np.expand_dims(img, axis = 0))

    ps = preds[0]
    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)
    classes = ['benign', 'malignant']
    print('Probabilities:')
    print(ps)
    print(pd.DataFrame({'Class Label': ['benign', 'malignant'], 'Probabilties': ps}))

    if ps[0] > 0.6:
        print("\nThis area of skin appears to be benign, but you should still check it out at the doctor's if you aren't sure!!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")
    else:
        print("\nThat's no ordinary abnormality, you should definitely check it out at the doctor's!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")

def prednowprednow(datagen, pred_model):
    images, labels = datagen.next()
    img = images[0]
    preds = model.predict(np.expand_dims(img, axis = 0))
    ps = preds[0]
    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)
    classes = ['benign', 'malignant']
    print('\n\nProbabilities:')
    print(ps)
    if ps[0] > 0.6:
        print("This area of skin appears to be benign, but you should still check it out at the doctor's if you aren't sure!!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")
    else:
        print("That's no ordinary abnormality, you should definitely check it out at the doctor's!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")

#pred_classify(prediction_generator, model)
prednowprednow(prediction_generator, model)
prednowprednow(prediction_generator, model)
prednowprednow(prediction_generator, model)
prednowprednow(prediction_generator, model)


print("\n\n\n\n\n\nIM DONE")
