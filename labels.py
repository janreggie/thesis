'''
labels.py

Opens a folder named /training/ and performs this.convert to said folder.
Writes out file data/training-labels.pkl.
'''
import os
import pickle
import glob
from os.path import join, exists


def convert(dataset):
    '''
    Opens the folder named dataset and returns a list.

    File structure:
    - /dataset
        - gesture1
            - file1.jpg
            - file2.jpg
        - gesture2
            - file1.jpg
            - file2.jpg

    Writes the following list to the file data/training-labels.pkl:
    [
        [/dataset/gesture1/file1.jpg, gesture1],
        [/dataset/gesture1/file2.jpg, gesture1],
        [/dataset/gesture2/file1.jpg, gesture2],
        [/dataset/gesture2/file2.jpg, gesture2],
    ]

    '''
    hc = []  # set as local variable cause it never really gets manip'd outside
    rootpath = os.getcwd()
    dataset = os.path.join(os.getcwd(), dataset)
    os.chdir(dataset)
    x = os.listdir(os.getcwd())

    for gesture in x:
        adhyan = gesture
        # path to the gesture itself (containing JPG's)
        gesture = os.path.join(dataset, gesture)
        os.chdir(gesture)

        for file in glob.glob('*.jpg'):
            hc.append([os.path.abspath(file), adhyan])
            print(adhyan)

    os.chdir(rootpath)

    with open('data/training-labels.pkl', 'wb') as handle:
        pickle.dump(hc, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    convert("training/")
