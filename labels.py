'''
labels.py

Opens a folder named /training/ and performs this.convert to said folder.
Writes out file data/training-labels.pkl.
'''
import os
import pickle
import glob
from PIL import Image
#from os.path import join, exists


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

    where the first argument is the image file in PIL.Image format!!
    '''
    hc = []  # set as local variable cause it never really gets manip'd outside
    rootpath = os.getcwd()
    os.chdir(dataset)
    x = os.listdir(os.getcwd())
    print(x)

    for gesture in x:
        adhyan = gesture
        # path to the gesture itself (containing JPG's)
        gesture = os.path.join(dataset, gesture)
        print(gesture)
        os.chdir(gesture)

        for file in glob.glob('**/*.jpg', recursive=True):
            print(file)
            hc.append([Image.open(file), adhyan])
            print(adhyan)

    os.chdir(rootpath)

    with open('training-labels.pkl', 'wb') as handle:
        pickle.dump(hc, handle, -1)


if __name__ == "__main__":
    convert(r"C:\Users\CTC219-PC01\Documents\thesis\bantupalli_xie\source_videos\train")
