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

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return str(int(n * multiplier) / multiplier)

def pred_classify(datagen, pred_model):
    # Get first batch
    images, labels = datagen.next()

    img = images[0]

    preds = model.predict(np.expand_dims(img, axis = 0))

    ps = preds[0]
    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)
    classes = ['benign', 'malignant']
    print(ps)

    print('Probabilities:')
    print(pd.DataFrame({'Class Label': ['benign', 'malignant'], 'Probabilties': ps}))

    if ps[0] > 0.6:
        print("This area of skin appears to be benign, but you should still check it out at the doctor's if you aren't sure!!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")
    else:
        print("That's no ordinary abnormality, you should definitely check it out at the doctor's!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")

def print_preds(datagen, pred_model):
    # Get first batch
    images, labels = datagen.next()

    img = images[0]
    preds = model.predict(np.expand_dims(img, axis = 0))
    ps = preds[0]

    print(img)
    # Swap the class name and its numeric representation
    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)

    print('Probabilities:')
    print(pd.DataFrame({'Class Label': classes, 'Probabilties': ps}))
    if labels[0][0] == 1.0:
        print('Actual Label: Benign')
    elif labels[0][1] == 1.0:
        print('Actual Label: Malignant')
    else:
        print('Actual Label: Label Error')
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.squeeze())
    ax1.axis('off')
    ax2.set_yticks(np.arange(len(classes)))
    ax2.barh(np.arange(len(classes)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticklabels(classes, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

IMAGE_WIDTH, IMAGE_HEIGHT = 200, 200
EPOCHS = 30
BATCH_SIZE = 32
num_classes = 2
'''
conda create -n tf tensorflow
conda activate tf
'''

def create_model():
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

#savedmodel = "/savemodel/skinpred.hdf5"
savedmodel = "/savemodel/skinpredv5.hdf5"
# v4 sucks, use v5

def lm():
    model = create_model()
    opt = Adam(lr = 0.0001)
    model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = opt, metrics = ['acc'])
    checkpointer = ModelCheckpoint(filepath="/savemodel/skinpred.hdf5", verbose=1, save_best_only=True)
    return model

model = lm()

fpath = ""
predlist = [fpath + "Melanoma.jpg", fpath + "Mole.jpg", fpath + "Skin.jpg", fpath + "Cancer.jpg", fpath + "Skin2.jpg"]
prediction_df = pd.DataFrame({'filename': predlist, 'class': ["malignant", "benign", "benign", "malignant", "benign"]})
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

    if ps[0] > 0.5:
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
    if ps[0] > 0.5:
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
prednowprednow(prediction_generator, model)


print("\nCODE COMPLETE")
