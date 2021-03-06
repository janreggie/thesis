"""
Utilities used by our other RNN scripts.
"""
from collections import deque
import math
import pickle
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
import tflearn
import numpy as np
import time


def get_data(filename, num_frames, num_classes, input_length, train):
    """Get the data from our saved predictions or pooled features.
    It has to load an object though... using make_predictions.py

    filename: object to load data from (iterable list with two items per object:
        probably features and label)
    num_frames: number of frames per video?
    num_classes: number of classifications? there could be
    input_length: similar to num_classes?
    """

    # Local vars.
    input_list = []
    output = []
    temp_list = deque()
    labels = {}
    count = 0

    # Open and get the features.
    with open(filename, 'rb') as fin:
        frames = pickle.load(fin)  # Open a filename into frames
        # for x in frames:
        #     print x

    for frame in frames:
        features = frame[0]
        # Possibly the name of the result itself?
        actual = frame[1].lower()

        # frameCount = frame[2]

        # Convert our labels into integers.
        if actual in labels:
            actual = labels[actual]
        else:
            labels[actual] = count
            # actual = count
            count += 1

        # Add to the lists.
        if len(temp_list) == num_frames - 1:
            temp_list.append(features)
            flat = list(temp_list)
            input_list.append(np.array(flat))
            output.append(actual)
            temp_list.clear()
        else:
            temp_list.append(features)
            continue

    for key in labels:
        print(key, labels[key])

    print("Total dataset size: %d" % len(input_list))

    # Numpy.
    input_list = np.array(input_list)
    output = np.array(output)

    print("\n", input_list.shape, output.shape, "\n")

    # Reshape.
    # Ignore too-many-function-args
    input_list = input_list.reshape(-1, num_frames, input_length)
    # Try to not think too much about it? It imports well...
    num_classes = len(labels)

    # print(X[1][0])

    print(num_classes)
    print(labels)

    # print(y)

    # One-hot encoded categoricals.
    output = to_categorical(output, num_classes)
    # y = y.reshape(-1, num_classes, input_length)

    # print(y)

    # Split into train and test.
    input_train, input_test, output_train, output_test = train_test_split(
        input_list, output, test_size=0.1)

    # num_of_rows = int((4) * 0.8)
    # num_of_rowz = int((80) * 0.8)
    #
    # # np.random.shuffle(X)
    # # np.random.shuffle(y)
    # X_train = X[:num_of_rows]
    # y_train = y[:num_of_rowz]
    #
    # num_of_rows = int(math.ceil((4) * 0.2))
    # num_of_rowz = int((80) * 0.2)
    #
    # X_test = X[:num_of_rows]
    # y_test = y[:num_of_rowz]
    #
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if train:
        return input_train, input_test, output_train, output_test
    # Otherwise..
    return input_list, output


def get_data_alt(input_file, output_file, num_frames, num_classes, input_length, train):
    """Homebrewed. Get the data from our saved predictions or pooled features.
    It has to load an object though... using make_predictions.py

    input_file: filename containing the features
    output_file: filename containing the labels
    num_frames: number of frames per video?
    num_classes: number of classifications? there could be
    input_length: similar to num_classes?
    """

    # Local vars.

    input_list = []
    output_list = []
    temp_list = deque()
    #classes = {}

    # Open and get the features.
    all_features = np.load(input_file)  # Open a filename into frames
    actual_labels = np.load(output_file)

    # print(all_features.shape[0])
    # print(actual_labels.shape[0])

    # time.sleep(10)

    for i in range(actual_labels.shape[0]):
        print("Opening {}".format(i))
        features = all_features[i]
        actual = actual_labels[i]
        # print(features)
        # print(actual)
        # Convert our classes into integers.
        # if actual in classes:
        #    actual = classes[actual]
        # else:
        #    classes[actual] = count
        #    actual = count
        #    count += 1

        # Add to the lists.
        if len(temp_list) == num_frames - 1:
            temp_list.append(features)
            flat = list(temp_list)
            input_list.append(np.array(flat))
            output_list.append(actual)
            temp_list.clear()
        else:
            temp_list.append(features)
            continue

    # for key in classes:
    #    print(key, classes[key])

    print("Total dataset size: %d" % len(input_list))

    # Numpy.
    input_list = np.array(input_list)
    output_list = np.array(output_list)

    print("Before reshaping: input:{}, output:{}".format(
        input_list.shape, output_list.shape))

    # Reshape.
    # Ignore too-many-function-args
    # input_list = input_list.reshape(-1, num_frames, input_length)
    # Try to not think too much about it? It imports well...
    #num_classes = len(classes)

    # print(X[1][0])

    # print(num_classes)
    # print(classes)

    # print(y)

    # One-hot encoded categoricals.
    # output_list = to_categorical(output_list, num_classes)
    # output_list = output_list.reshape(-1, num_classes, input_length)

    # print(y)

    # Split into train and test.
    print("After reshaping: input:{}, output:{}".format(
        input_list.shape, output_list.shape))
    # output_list = output_list.flatten()
    # print(input_list[0])
    # print(output_list[0])
    time.sleep(10)
    print("After flattening: input:{}, output:{}".format(
        input_list.shape, output_list.shape))
    input_train, input_test, output_train, output_test = train_test_split(
        input_list, output_list, test_size=0.1)

    # num_of_rows = int((4) * 0.8)
    # num_of_rowz = int((80) * 0.8)
    #
    # # np.random.shuffle(X)
    # # np.random.shuffle(y)
    # X_train = X[:num_of_rows]
    # y_train = y[:num_of_rowz]
    #
    # num_of_rows = int(math.ceil((4) * 0.2))
    # num_of_rowz = int((80) * 0.2)
    #
    # X_test = X[:num_of_rows]
    # y_test = y[:num_of_rowz]
    #
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if train:
        return input_train, input_test, output_train, output_test
    # Otherwise..
    return input_list, output_list


def get_network(frames, input_size, num_classes):
    """Create an LSTM network of two 128-unit layers"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 128, dropout=0.8, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_deep(frames, input_size, num_classes):
    """Create a deeper LSTM of three 64-unit layers"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_wide(frames, input_size, num_classes):
    """Create a wider LSTM of one 256-unit layer"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    # net = tflearn.input_data(shape=[None, frames])
    net = tflearn.lstm(net, 256, dropout=0.3)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net


def get_network_wider(frames, input_size, num_classes):
    """Create a much wider LSTM of one 512-unit layer"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 512, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net
