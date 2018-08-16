import pickle
import sys
import tensorflow as tf
from tqdm import tqdm


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander,  [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def get_labels(label_file):
    """Return a list of our trained labels so we can
    test our training accuracy. The file is in the
    format of one label per line, in the same order
    as the predictions are made. The order can change
    between training runs."""
    label = []
    lines = tf.gfile.GFile(label_file).readlines()
    for i in lines:
        label.append(i.rstrip())

    return label


def predict_on_frames(frames, batch):
    """Given a list of frames, predict all their classes."""
    # Unpersists graph from file

    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"

    graph = load_graph(model_file)

    frames = get_labels(label_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    frame_predictions = []

    for i, frame in enumerate(frames):
        filename = frame[0]
        label = frame[1]

        t = read_tensor_from_image_file(filename)

        with tf.Session() as sess:
            pbqr = tqdm(total=len(frames))

            try:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })

                prediction = results[0]
            except KeyboardInterrupt:
                print("You chose to exit.")
                sys.exit()

            frame_predictions.append([prediction, label])

            if i > 0 and i % 10 == 0:
                pbar.update(10)

        pbar.close()
        return frame_predictions



def get_accuracy(predictions, labels):
    """After predicting on each batch, check that batch's
    accuracy to make sure things are good to go. This is
    a simple accuracy metric, and so doesn't take confidence
    into account, which would be a better metric to use to
    compare changes in the model."""
    correct = 0
    for frame in predictions:
        # Get the highest confidence class.
        this_prediction = frame[0].tolist()
        # print this_prediction
        this_label = frame[1]
        # print this_label

        max_value = max(this_prediction)
        max_index = this_prediction.index(max_value)
        predicted_label = labels[max_index]
        # print predicted_label

        # Now see if it matches.
        print (predicted_label, this_label)
        if predicted_label.lower() == this_label.lower():
            correct += 1
        print (correct, len(predictions))

    print (correct, len(predictions))
    accuracy = correct / float(len(predictions))
    return accuracy


def main():
    labels = get_labels()
    # print labels
    batch = '1'
    # batch = '2'

    with open('data/training-labels.pkl', 'rb') as fin:
        frames = pickle.load(fin)

    # for frame in frames:
    #     print frame
    # Predict on this batch and get the accuracy.
    predictions = predict_on_frames(frames, batch)
    for frame in predictions:
        print (frame)
    accuracy = get_accuracy(predictions, labels)
    print("Batch accuracy: %.5f" % accuracy)

    # Save it.
    with open('data/predicted-frames-' + batch + '.pkl', 'wb') as fout:
        pickle.dump(predictions, fout)

    # for batch in batches:
    #     print("Doing batch %s" % batch)
    #     with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
    #         frames = pickle.load(fin)

    #     # Predict on this batch and get the accuracy.
    #     predictions = predict_on_frames(frames, batch)
    #     accuracy = get_accuracy(predictions, labels)
    #     print("Batch accuracy: %.5f" % accuracy)

    #     # Save it.
    #     with open('data/predicted-frames-' + batch + '.pkl', 'wb') as fout:
    #         pickle.dump(predictions, fout)
    print("Done.")

if __name__ == '__main__':
    main()
