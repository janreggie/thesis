import os
import pickle
import glob
from os.path import join, exists
hc = []
frameCount = 60

def convert(dataset):
    rootpath = os.getcwd()
    dataset = os.path.join(os.getcwd(), dataset)
    os.chdir(dataset)
    x = os.listdir(os.getcwd())

    for gesture in x:
        adhyan = gesture
        gesture = os.path.join(dataset, gesture)
        os.chdir(gesture)

        for file in glob.glob('*.jpg'):
            hc.append([os.path.abspath(file), adhyan])
            print(adhyan)

    os.chdir(rootpath)
    os.chdir(rootpath)

    with open('data/training-labels.pkl', 'wb') as handle:
        pickle.dump(hc, handle, protocol=pickle.HIGHEST_PROTOCOL)


convert("training/")
