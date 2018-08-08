from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
import numpy as np
import glob, os
from scipy.misc import imresize, imread

batch_size = 128


def bring_data_from_directory():
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_directory('train',
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  classes=['again', 'bad', 'become', 'beer'])
    validation_generator = datagen.flow_from_directory('validate',
                                                       target_size=(224, 224),
                                                       batch_size=128,
                                                       class_mode='categorical',
                                                       shuffle=True,
                                                       classes=['again', 'bad', 'become', 'beer'])

    return train_generator, validation_generator


def load_VGG16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print('Model loaded!')
    print(base_model.summary())
    return base_model


def extract_features_and_store(train_generator, validation_generator, base_model):
    x_generator = None
    y_label = None

    batch = 0
    for x, y in train_generator:
        if batch == (56021/batch_size):
            break
        print("Predict on batch: ", batch)
        batch += 1
        if x_generator==None:
            x_generator = base_model.predict(x)
            y_label = y
            print(y)
        else:
            x_generator = np.append(x_generator, base_model.predict(x), axis=0)
            y_label = np.append(y_label, y, axis=0)

    x_generator, y_label = shuffle(x_generator, y_label)

    np.save(open('video_x_VGG16.npy', 'w'), x_generator)
    np.save(open('video_y_VGG16.npy', 'w'), y_label)

    batch = 0

    x_generator = None
    y_label = None

    for x, y in validation_generator:
        if batch == (3974/batch_size):
            break
        print ("Predict on batch validate:", batch)
        batch += 1
        if x_generator == None:
            x_generator = base_model.predict(x)
            y_label = y
        else:
            x_generator = np.append(x_generator, base_model.predict(x), axis=0)
            y_label = np.append(y_label, y, axis=0)

    x_generator, y_label = shuffle(x_generator, y_label)
    np.save(open('video_x_validate_VGG16.npy', 'w'), x_generator)
    np.save(open('video_y_validate_VGG16.npy', 'w'), y_label)

    train_data = np.load(open('video_x_VGG16.npy'))
    train_labels = np.load(open('video_y_VGG16.npy'))
    train_data, train_labels = shuffle(train_data, train_labels)
    validation_data = np.load(open('video_x_validate_VGG16.npy'))
    validation_labels = np.load(open('video_y_validate_VGG16.npy'))
    validation_data, validation_labels = shuffle(validation_data, validation_labels)

    return train_data, train_labels, validation_data, validation_labels





if __name__ == '__main__':
    train_generator, validation_generator = bring_data_from_directory()
    base_model = load_VGG16_model()
    train_data, train_labels, validation_data, validation_labels = extract_features_and_store(train_generator,
                                                                                              validation_generator, base_model)