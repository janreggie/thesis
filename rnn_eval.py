"""
Run a holdout set of data through our trained RNN. Requires we first
run train_rnn.py and save the weights.
"""
from rnn_utils import get_network, get_network_deep, get_network_wide, get_data
import tflearn
import numpy as np


def main(filename, frames, batch_size, num_classes, input_length):
    """From the blog post linked above."""
    # Get our data.
    X_train, y_train = get_data(
        filename, frames, num_classes, input_length, False)
    # print X_train
    # print y_train

    # Get sizes.
    # print ("Y train :- ", y_train)
    print("X_train:-", X_train[0])
    num_classes = len(y_train[0])

    # Get our network.
    net = get_network_wide(frames, input_length, num_classes)

    # Get our model.
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load('checkpoints/rnn.tflearn')
    # model.load('checkpoints_backup/rnn.tflearn')

    # Evaluate.
    hc = model.predict(X_train)
    hc = [np.argmax(every) for every in hc]
    print(hc)
    aadi = [np.argmax(every) for every in y_train]
    print("l1 :", len(aadi))
    print("l2 ", len(hc))
    answer = []

    for i in range(0, len(hc)):
        answer.append([aadi[i], hc[i]])

    answer.sort()
    print(answer)
    f = open("results.txt", "w")
    for x in answer:
        print(x[0], x[1])
        f.write(str(x[0])+" "+str(x[1])+"\n")

    print(model.evaluate(X_train, y_train))


if __name__ == '__main__':
    FILENAME = 'data/test-results.pkl'
    INPUT_LENGTH = 4
    # FILENAME = 'data/cnn-features-frames-2.pkl'
    # INPUT_LENGTH = 2048
    FRAMES = 118
    BATCH_SIZE = 64
    NUM_CLASSES = 4

    main(FILENAME, FRAMES, BATCH_SIZE, NUM_CLASSES, INPUT_LENGTH)
