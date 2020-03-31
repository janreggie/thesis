'''
make_predictions.py

Makes predictions out of a set of frames (train-labels.pkl).

Opens data/train-labels.pkl, retrained_labels.txt, and retrained_graph.pb
and dumps data to data/predicted-training-frames.pkl
'''
import pickle
import sys
import tensorflow as tf
from tqdm import tqdm


def get_labels():
    '''
    Returns a list of strings containing the labels.
    '''
    with open('retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels


def predict_on_frames(frames):
    '''
    To be used for training the LSTM neural net.
    Returns a list encoding the "predictions" of individual frames
    (what they seem to be) as well as their actual labels.

    Requires a retrained_graph.pb.
    '''
    with tf.gfile.FastGFile('retrained_graph.pb', 'rb') as fin:
        graph_def = tf.GraphDef()  # Serialized dataflow Graph
        graph_def.ParseFromString(fin.read())
        _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        frame_predictions = []
        pbar = tqdm(total=len(frames))

        for i, frame in enumerate(frames):
            filename = frame[0]
            label = frame[1]

            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            try:
                predictions = sess.run(
                    softmax_tensor,
                    {'DecodeJpeg/contents:0': image_data}
                )
                prediction = predictions[0]
            except KeyboardInterrupt:
                print("You quit with Ctrl + C")
                sys.exit()
            except:
                print("Error")
                continue

            frame_predictions.append([prediction, label])

            if i > 0 and i % 10:
                pbar.update(10)
        pbar.close()

        return frame_predictions


def get_accuracy(predictions, labels):
    '''
    Determines the accuracy of the predictions provided. 
    Returns a single float representing accuracy.

    predictions: from predict_on_frames.
    labels: list of labels (get_labels)
    '''
    correct = 0
    for frame in predictions:
        this_prediction = frame[0].tolist()
        this_label = frame[1]

        max_value = max(this_prediction)
        max_index = this_prediction.index(max_value)
        predicted_label = labels[max_index]

        if predicted_label == this_label:
            correct += 1

    accuracy = correct / len(predictions)
    return accuracy


def main():
    '''
    Does something
    '''
    # batches = ['1']
    labels = get_labels()

    with open('data/train-labels.pkl', 'rb') as fin:
        frames = pickle.load(fin)

    predictions = predict_on_frames(frames)
    accuracy = get_accuracy(predictions, labels)
    print("Accuracy : ", accuracy)

    with open('data/predicted-training-frames.pkl', 'wb') as fout:
        pickle.dump(predictions, fout)

    print("Done")


if __name__ == '__main__':
    main()
