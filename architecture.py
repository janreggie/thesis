'''
architecture.py

The neural net architecture used.
The first few are convolutional, the next few are fully-connected neurons,
and the last few are recurrent.
'''
import tflearn

TEST_FOLDER = 'test'
TRAIN_FOLDER = 'training'


def getXY():
    '''
    Loads TRAIN_FOLDER and returns inputs and outputs
    '''
    x, y = tflearn.data_utils.image_preloader(TRAIN_FOLDER, image_shape=(
        80, 80), mode='folder', categorical_labels='True', normalize=True)
    return x, y


def main():
    '''
    Establishes architecture for CNN-fully-connected-RNN neural net.
    '''
    # Create the input layer: None: batch size, 120: frames, 80: height, 80: width, 3: RGB
    network = tflearn.input_data(shape=[None, 120, 80, 80, 3], name='input')

    # Convolutional network
    network = tflearn.conv_3d(
        network, 32, (3, 3, 3), activation='relu')  # 32 conv layers of 3x3x3 (3x3x3 convolves for each 3 frames, 3px height, and 3px width)
    network = tflearn.max_pool_3d(
        network, (1, 2, 2), strides=(1, 2, 2))  # Pools results of the conv_3d layer
    network = tflearn.conv_3d(
        network, 64, (3, 3, 3), activation='relu')  # 64 layers of 3x3x3
    network = tflearn.max_pool_3d(network, (1, 2, 2), strides=(1, 2, 2))
    network = tflearn.conv_3d(
        network, 128, (3, 3, 3), activation='relu')  # 128 layers of 3x3x3
    network = tflearn.conv_3d(
        network, 128, (3, 3, 3), activation='relu')  # another one?
    network = tflearn.max_pool_3d(network, (1, 2, 2), strides=(1, 2, 2))
    network = tflearn.conv_3d(
        network, 256, (2, 2, 2), activation='relu')  # 256 layers of 2x2x2
    network = tflearn.conv_3d(
        network, 256, (2, 2, 2), activation='relu')
    network = tflearn.max_pool_3d(
        network, (1, 2, 2), strides=(1, 2, 2))
    network = tflearn.conv_2d(
        network, 64, 4, activation='relu', regularizer="L2")  # 64 layers of 4x4
    network = tflearn.max_pool_2d(network, 2)  # and then max pool

    # Normalize activations of the previous layer at each batch.
    network = tflearn.local_response_normalization(network)

    # And now the fully-connected neural net (128 & 256 neurons + dropout)
    network = tflearn.fully_connected(network, 128, activation='tanh')
    network = tflearn.dropout(network, 0.8)
    network = tflearn.fully_connected(network, 256, activation='tanh')
    network = tflearn.dropout(network, 0.8)
    network = tflearn.reshape(network, [-1, 1, 256])  # Why 256?

    # LSTM layers
    network = tflearn.lstm(network, 128, return_seq=True)  # LSTM of 128 units
    network = tflearn.lstm(network, 128)
    network = tflearn.fully_connected(
        network, 4, activation='softmax')  # Just four neurons... okay?
    network = tflearn.regression(
        network, optimizer='adam', loss='categorical_crossentropy', name='target')

    # Tries to fit TRAIN_FOLDER
    model = tflearn.DNN(network, tensorboard_verbose=0)
    X, Y = getXY()
    model.fit(X, Y, n_epoch=1, validation_set=0.1,
              show_metric=True, snapshot_step=100)


if __name__ == "__main__":
    main()
