"""
Given a saved output of predictions or pooled features from our CNN,
train an RNN (LSTM) to examine temporal dependencies.
"""
from rnn_utils import get_network, get_network_deep, get_network_wide, get_data, get_network_wider
import os
import tflearn
from tflearn.metrics import Accuracy


def main(filename, frames, batch_size, num_classes, input_length):
    """From the blog post linked above."""
    # Get our data.
    X_train, X_test, y_train, y_test = get_data(
        filename, frames, num_classes, input_length, True)

    # Get sizes.
    print ("y_train[0] : ", y_train)
    num_classes = len(y_train[0])
    print ("Num classes : - ", num_classes)
    # print "Y train : ", y_train[0]

    # Get our network.
    # net = get_network_deep(frames, input_length, num_classes)
    net = get_network_deep(frames, input_length, num_classes)

    # Train the model.
    if os.path.exists('checkpoints/rnn.tflearn'):
        print ("Model already exists! Loading it")
        model.load('checkpoints/rnn.tflearn')
        print ("Model Loaded")
    else:
        model = tflearn.DNN(net, max_checkpoints=1, tensorboard_verbose=0)

    model.fit(X_train, y_train, validation_set=(X_test, y_test),
              show_metric=True, batch_size=batch_size, snapshot_step=100,
              n_epoch=10, run_id='name_model')


    # Save it.
    x = input("Do you wanna save the model and overwrite? y or n")
    if(x == "y"):
        model.save('checkpoints/rnn.tflearn')

if __name__ == '__main__':
    # filename = 'data/cnn-features-frames-1.pkl'
    # input_length = 2048
    filename = 'data/training-results.pkl'
    input_length = 4
    frames = 118
    batch_size = 64
    num_classes = 4

    main(filename, frames, batch_size, num_classes, input_length)
