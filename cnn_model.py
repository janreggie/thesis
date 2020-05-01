'''
cnn_model.py

Folder structure
train/
    again/
        file1.jpg
        file2.jpg
        ...
    bad/
        file1.jpg
        file2.jpg
        ...
    become/
        ...
    beer/
        ...
validate/
    again/
        ...
    bad/
        ...
    become/
        ...
    beer/
        ...

Ellipses and file{i}.jpg indicate image files
that aren't necessarily JPEGs.
'''
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from sklearn.utils import shuffle
import numpy as np

BATCH_SIZE = 128
TRAIN_FOLDER = 'train'
VALIDATE_FOLDER = 'validate'
SOURCE_FOLDER = 'C:/Users/ryZen/Downloads/Thesis/source_videos/'


def bring_data_from_directory():
    '''
    Generate batches of augmented training and validation data.
    '''
    # ImageGenerator generates batches of tensor image data with real-time data
    # augmentation. Looped over in batches.
    # rescale=1./255 means scale image to floating point 0<=i<=1.
    datagen = ImageDataGenerator(rescale=1. / 255)

    # Generators that open TRAIN_FOLDER and VALIDATE_FOLDER's images
    # that are resized to 224x224 and batched
    train_generator = datagen.flow_from_directory(
        TRAIN_FOLDER,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=['again', 'bad', 'become', 'beer'])
    validation_generator = datagen.flow_from_directory(
        VALIDATE_FOLDER,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=['again', 'bad', 'become', 'beer'])

    return train_generator, validation_generator


def load_VGG16_model():
    '''
    Generate a VGG16 neural net model with weights pre-trained on ImageNet.
    Input shape is size of the input image (224x224px, 3 channels).
    Output shape is a Tensor(?) containing the "features" of the input.
    '''
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    print('Model loaded!')
    print(base_model.summary())
    return base_model


def load_inception_model():
    '''Loads a Inception V3 neural net model
    '''
    base_model = InceptionV3(weights='imagenet', include_top=False,
                             input_shape=(224, 224, 3))
    print(base_model.summary())
    return base_model


def extract_features_and_store(train_generator, validation_generator, base_model):
    '''
    Extracts features from train and validation generators (bring_data_from_directory)
    using a base_model and returns the input (features) and output (labels) tensors.

    Files that will be written:
    - video_x_train_inception.npy: input tensors for training data
    - video_y_train_inception.npy: output labels for training data
    - video_x_validate_inception.npy: input tensors for validation data
    - video_y_validate_inception.npy: output labels for validation data
    '''
    # Extract features from input, output data in train_generator
    input_tensors = None   # "features" according to the CNN
    output_tensors = None  # the actual labels (classes)
    batch = 0
    for input_datum, output_datum in train_generator:
        # input_datum: numpy array of batch of images
        # output_datum: numpy array of corresponding labels
        if batch >= (7948/BATCH_SIZE):  # where did 56021 come from?!
            break
        print("Predict on batch: ", batch)
        if input_tensors is None:
            input_tensors = base_model.predict(input_datum)
            output_tensors = output_datum
            # print(output_datum)
        else:
            input_tensors = np.append(
                input_tensors, base_model.predict(input_datum), axis=0)
            output_tensors = np.append(output_tensors, output_datum, axis=0)
        batch += 1

    # After that, they are to be shuffled and persistently saved.
    input_tensors, output_tensors = shuffle(input_tensors, output_tensors)
    np.save('video_x_train_inception.npy', input_tensors)
    np.save('video_y_train_inception.npy', output_tensors)

    # Now do the same for validation_generator
    input_tensors = None
    output_tensors = None
    batch = 0
    for input_datum, output_datum in validation_generator:
        if batch >= (3974/BATCH_SIZE):  # where did 3974 come from?!
            break
        print("Predict on batch validate:", batch)
        if input_tensors is None:
            input_tensors = base_model.predict(input_datum)
            output_tensors = output_datum
        else:
            input_tensors = np.append(
                input_tensors, base_model.predict(input_datum), axis=0)
            output_tensors = np.append(output_tensors, output_datum, axis=0)
        batch += 1

    input_tensors, output_tensors = shuffle(input_tensors, output_tensors)
    np.save('video_x_validate_inception.npy', input_tensors)
    np.save('video_y_validate_inception.npy', output_tensors)

    train_data = np.load('video_x_train_inception.npy')
    train_labels = np.load('video_y_train_inception.npy')
    train_data, train_labels = shuffle(train_data, train_labels)
    validation_data = np.load('video_x_validate_inception.npy')
    validation_labels = np.load('video_y_validate_inception.npy')
    validation_data, validation_labels = shuffle(
        validation_data, validation_labels)

    return train_data, train_labels, validation_data, validation_labels


def main():
    '''
    Opens files from directories and creates training and validation input and
    output tensors, saving them to disk and printing them out.
    '''
    train_generator, validation_generator = bring_data_from_directory()
    base_model = load_inception_model()
    train_data, train_labels, validation_data, validation_labels = extract_features_and_store(
        train_generator,
        validation_generator, base_model)
    # Now what....
    # Well... we can print them out I guess
    print("Training data: {}".format(train_data))
    print("Training labels: {}".format(train_labels))
    print("Validation data: {}".format(validation_data))
    print("Validation labels: {}".format(validation_labels))


if __name__ == '__main__':
    os.chdir(SOURCE_FOLDER)
    main()
