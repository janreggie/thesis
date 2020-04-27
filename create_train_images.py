'''
create_train_images.py

Refer to split for documentation
'''
import os
import numpy as np
import cv2
from imageio import imread, imresize


def split(path):
    '''
    Returns two lists: a list of all images it can find in path,
    and their corresponding gesture_folder index.
    Resizes all images to 224x224.
    Selects every four images in the gesture_folder named "beer"
    and every eight in other gestures.
    Also writes every four/eight images to some file...

    File structure:
    - path
        - videoclass1
            - gesture1
                - file{1-100}.jpg  # 1 to 100 inclusive
            - gesture2
                - file{1-100}.jpg
        - videoclass2
            - ...

    Returns:
    input_list = [
        path/videoclass1/gesture1/file{8,16,24,...,96}.jpg, 
        path/videoclass1/gesture2/file{4,12,20,...,100}.jpg]
    output_list = [
        0,0,0,...,0,
        1,1,1,...,1]

    Writes:
    path/videoclass1/gesture1/file{8,16,24,...,96}.jpg -> path/videoclass1/{0,1,2,...,11}_file{8,16,24,...,96}.jpg
    path/videoclass1/gesture2/file{4,12,20,...,100}.jpg -> path/videoclass1/{12,13,...,24}_file{4,12,20,...,100}.jpg
    '''
    input_list = []
    output_list = []
    count = 0

    # Choose path as directory
    # rootpath = os.getcwd()
    # dataset = os.path.join(os.getcwd(), path)
    owd = os.getcwd()
    os.chdir(path)
    dataset = os.getcwd()

    for video_class in os.listdir(dataset):
        print(video_class)
        gesture_folders = os.listdir(os.path.join(dataset, video_class))

        for gesture_index, gesture_folder in enumerate(gesture_folders):
            gesture = os.listdir(
                os.path.join(dataset, video_class, gesture_folder))

            for count, image_filename in enumerate(gesture):
                if video_class == 'beer':  # Somehow this is a different case...?
                    if count % 4 == 0:
                        image = imread(
                            os.path.join(dataset, video_class, gesture_folder, image_filename))
                        image = imresize(image, (224, 224))
                        input_list.append(image)
                        output_list.append(gesture_index)
                        cv2.imwrite(
                            os.path.join(dataset, video_class, str(count)+'_'+image_filename), image)

                else:
                    if count % 8 == 0:
                        image = imread(
                            os.path.join(dataset, video_class, gesture_folder, image_filename))
                        image = imresize(image, (224, 224))
                        input_list.append(image)
                        output_list.append(gesture_index)
                        cv2.imwrite(
                            os.path.join(dataset, video_class, str(count)+'_'+image_filename), image)

    input_list = np.array(input_list)
    output_list = np.array(output_list)
    os.chdir(owd)

    return input_list, output_list


if __name__ == '__main__':
    split("training/")
