import tflearn
import tensorflow as tf
import numpy as np
from tflearn.data_utils import image_preloader
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

test = 'test'
train = 'training'

def getXY():
    X, Y = image_preloader(train, image_shape=(80, 80), mode='folder', categorical_labels='True', normalize=True)
    return X, Y

network = tflearn.input_data(shape=[None, 120, 80, 80, 3], name='input')
network = tflearn.conv_3d(network, 32, (3, 3, 3), activation='relu')
network = tflearn.max_pool_3d(network, (1, 2, 2), strides=(1, 2, 2))
network = tflearn.conv_3d(network, 64, (3, 3, 3), activation='relu')
network = tflearn.max_pool_3d(network, (1, 2, 2), strides=(1, 2, 2))
network = tflearn.conv_3d(network, 128, (3, 3, 3), activation='relu')
network = tflearn.conv_3d(network, 128, (3, 3, 3), activation='relu')
network = tflearn.max_pool_3d(network, (1, 2, 2), strides=(1, 2, 2))
network = tflearn.conv_3d(network, 256, (2, 2, 2), activation='relu')
network =tflearn.conv_3d(network, 256, (2, 2, 2), activation='relu')
network = tflearn.max_pool_3d(network, (1, 2, 2), strides=(1, 2, 2))



network = tflearn.conv_2d(network, 64, 4, activation='relu', regularizer="L2")
network = tflearn.max_pool_2d(network, 2)
network = tflearn.local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = tflearn.reshape(network, [-1, 1, 256])

network = tflearn.lstm(network, 128, return_seq=True)
network = tflearn.lstm(network, 128)
network = tflearn.fully_connected(network, 4, activation='softmax')
network = tflearn.regression(network, optimizer='adam', loss='categorical_crossentropy',
                             name='target')

model = tflearn.DNN(network, tensorboard_verbose=0)
X, Y = getXY()
model.fit(X, Y, n_epoch=1, validation_set=0.1, show_metric=True, snapshot_step=100)