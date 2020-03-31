import os
import glob
import cv2
from imageio import imread, imresize
import numpy as np


def split(path):
    x = []
    y = []
    count = 0
    output = 0

    owd = os.getcwd()
    os.chdir(path)

    for video_class in os.listdir(os.getcwd()):
        print(video_class)
        gesture = os.listdir(os.path.join(path, video_class))
        for class_i in gesture:
            sub_child = os.listdir(os.path.join(path, video_class, class_i))
            for file in sub_child:
                if video_class == 'beer':
                    if count % 4 == 0:
                        image = imread(
                            os.path.join(path, video_class, class_i, file))
                        image = imresize(image, (224, 224))
                        x.append(image)
                        y.append(output)
                        cv2.imwrite(
                            os.path.join(path, video_class, str(count) + '_' + file), image)

                else:
                    if count % 8 == 0:
                        image = imread(
                            os.path.join(path, video_class, class_i, file))
                        image = imresize(image, (224, 224))
                        x.append(image)
                        y.append(output)
                        cv2.imwrite(path + '/' + video_class +
                                    '/' + str(count) + '_' + file, image)
                count += 1
            output += 1
    x = np.array(x)
    y = np.array(y)
    print("x", len(x), "y", len(y))
    os.chdir(owd)


if __name__ == '__main__':
    split("testset/")
