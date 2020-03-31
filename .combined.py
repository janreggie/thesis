'''
Doesn't do anything either really...
'''
import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d, max_pool_2d

network = input_data(shape=[None, frames, input_size])
net = conv_2d(network, 64, 7, activation='relu')
