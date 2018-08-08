"""
Utilities used by our other RNN scripts.
"""
from collections import deque
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
import tflearn
import numpy as np
import pickle
import math


def get_data(filename, num_frames, num_classes, input_length, ifTrain):
    """Get the data from our saved predictions or pooled features."""

    # Local vars.
    X = []
    y = []
    temp_list = deque()
    labels = {}
    count = 0

    # Open and get the features.
    with open(filename, 'rb') as fin:
        frames = pickle.load(fin)
        # for x in frames:
        #     print x
        for i, frame in enumerate(frames):

            features = frame[0]
            actual = frame[1].lower()

            # frameCount = frame[2]

            # Convert our labels into binary.
            if actual in labels:
                actual = labels[actual]
            else:
                labels[actual] = count
                # actual = count
                count += 1

            # Add to the queue.
            if len(temp_list) == num_frames - 1:
                temp_list.append(features)
                flat = list(temp_list)
                X.append(np.array(flat))
                y.append(actual)
                temp_list.clear()
            else:
                temp_list.append(features)
                continue
    for key in labels:
        print(key, labels[key])

    print("Total dataset size: %d" % len(X))

    # Numpy.
    X = np.array(X)
    y = np.array(y)

    print("\n", X.shape, y.shape, "\n")

    # Reshape.
    X = X.reshape(-1, num_frames, input_length)
    num_classes = len(labels)

    # print(X[1][0])

    print (num_classes)
    print (labels)

    # print(y)

    # One-hot encoded categoricals.
    y = to_categorical(y, num_classes)
    # y = y.reshape(-1, num_classes, input_length)

    # print(y)

    # Split into train and test.
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.1, random_state=42)

    X_train, X_test = train_test_split(X, test_size=0.1)
    y_train, y_test = train_test_split(y, test_size=0.1)

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

    if ifTrain:
        return X_train, X_test, y_train, y_test
    else:
        return X, y


def get_network(frames, input_size, num_classes):
    """Create our LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 128, dropout=0.8, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_deep(frames, input_size, num_classes):
    """Create a deeper LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_wide(frames, input_size, num_classes):
    """Create a wider LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    # net = tflearn.input_data(shape=[None, frames])
    net = tflearn.lstm(net, 256, dropout=0.3)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net


def get_network_wider(frames, input_size, num_classes):
    """Create a wider LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 512, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net
