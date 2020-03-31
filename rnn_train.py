"""
Given a saved output of predictions or pooled features from our CNN,
train an RNN (LSTM) to examine temporal dependencies.
"""
import os
import tflearn
from rnn_utils import get_network_deep, get_data


def main(filename, frames, batch_size, num_classes, input_length):
    """From the blog post linked above."""
    # Get our data.
    x_train, x_test, y_train, y_test = get_data(
        filename, frames, num_classes, input_length, True)

    # Get sizes.
    print("y_train[0] : ", y_train)
    num_classes = len(y_train[0])
    print("Num classes : - ", num_classes)
    # print "Y train : ", y_train[0]

    # Get our network.
    # net = get_network_deep(frames, input_length, num_classes)
    net = get_network_deep(frames, input_length, num_classes)

    # Train the model.
    if os.path.exists('checkpoints/rnn.tflearn'):
        print("Model already exists! Loading it")
        model = tflearn.DNN(net)
        model.load('checkpoints/rnn.tflearn')
        print("Model Loaded")
    else:
        model = tflearn.DNN(net, max_checkpoints=1, tensorboard_verbose=0)

    model.fit(x_train, y_train, validation_set=(x_test, y_test),
              show_metric=True, batch_size=batch_size, snapshot_step=100,
              n_epoch=10, run_id='name_model')

    # Save it.
    print('Saving to checkpoints/rnn.tflearn')
    model.save('checkpoints/rnn.tflearn')


if __name__ == '__main__':
    # filename = 'data/cnn-features-frames-1.pkl'
    # input_length = 2048
    # FILENAME = 'data/training-results.pkl'
    FILENAME = 'video_x_InceptionV3.npy'
    INPUT_LENGTH = 4
    FRAMES = 118
    BATCH_SIZE = 64
    NUM_CLASSES = 4

    main(FILENAME, FRAMES, BATCH_SIZE, NUM_CLASSES, INPUT_LENGTH)
