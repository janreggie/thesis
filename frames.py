import os
import glob
import cv2
import math


def videos():
    count = 1
    for file in glob.glob('*.mov'):
        video = cv2.VideoCapture(file)
        print(video.isOpened())
        framerate = video.get(5)
        os.makedirs("C:/Users/kbantupa/Desktop/new/" + "video_" + str(int(count)))
        while video.isOpened():
            frameId = video.get(1)
            success, image = video.read()
            if image is not None:
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            if success is not True:
                break
            if frameId % math.floor(framerate) == 0:
                filename = "C:/Users/kbantupa/Desktop/new/video_" + str(int(count)) + "/image" + \
                           str(int(frameId / math.floor(framerate)) + 1) + ".jpg"
                print(filename)
                cv2.imwrite(filename, image)
        video.release()
        print('done')
        count += 1


if __name__ == '__main__':
    os.chdir("C://Users//kbantupa//Desktop//new//bad")
    videos()
