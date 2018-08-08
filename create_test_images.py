import numpy as np
import os
import glob
import pandas as pd
import cv2
from scipy.misc import imread, imresize
import pickle


x = []
y = []
count = 0
output = 0


def split(path):
    global count
    global output
    global x
    global y

    rootpath = os.getcwd()
    dataset = os.path.join(os.getcwd(), path)
    os.chdir(dataset)

    for video_class in os.listdir(os.getcwd()):
        print(video_class)
        gesture = os.listdir(dataset + "/" + video_class)
        for class_i in gesture:
            sub_child = os.listdir(dataset + "/" + video_class + "/" + class_i)
            for file in sub_child:
                if video_class == 'beer':
                    if count % 4 == 0:
                        image = imread(dataset + "/" + video_class + "/" + class_i + "/" + file)
                        image = imresize(image, (224, 224))
                        x.append(image)
                        y.append(output)
                        cv2.imwrite(dataset + '/' + video_class + '/' + str(count) + '_' + file, image)
                    count += 1

                else:
                    if count % 8 == 0:
                        image = imread(dataset + "/" + video_class + "/" + class_i + "/" + file)
                        image = imresize(image, (224, 224))
                        x.append(image)
                        y.append(output)
                        cv2.imwrite(dataset + '/' + video_class + '/' + str(count) + '_' + file, image)
                    count += 1
            output += 1
    x = np.array(x)
    y = np.array(y)
    print("x", len(x), "y", len(y))


if __name__ == '__main__':
    split("testset/")
