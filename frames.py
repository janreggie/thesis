'''
frames.py

Does something I guess
'''
import os
import glob
import math
import cv2


def videos(directory):
    '''
    looks for all videos in directory and writes every FRAMERATE frames
    to directory "video_count" for count = 1 to how many files there are in directory
    that end with ".mov".

    Directory:
    /directory
        /aleph.mov
        /beth.mov

    Writes:
    /directory
        /video_1        # aleph.mov
            /1.jpg
            /2.jpg
            /3.jpg      # for every framerate frames in aleph.mov
        /video_2        # beth.mov
            /1.jpg
            /2.jpg
            /3.jpg
    '''
    owd = os.getcwd()  # return later
    os.chdir(directory)

    for count, file in enumerate(glob.glob('*.mov'), start=1):
        video = cv2.VideoCapture(file)  # reads file.mov
        print(video.isOpened())
        framerate = video.get(cv2.CAP_PROP_FPS)
        # video_dir =  video_1, video_2, ... for every mov thingy
        video_dir = "video_" + str(count)
        os.makedirs(os.path.join(directory, video_dir), exist_ok=True)
        while video.isOpened():
            frame_id = video.get(cv2.CAP_PROP_POS_FRAMES)
            success, image = video.read()  # grabs, decodes, reads next video frame
            if success is not True:
                break
            if image is not None:  # resize to 224x224
                # Resize to only top of image (resize by half)
                dimensions = image.shape
                image = image[:dimensions[0]//2, :dimensions[1]]
                image = cv2.resize(image, (224, 224),
                                   interpolation=cv2.INTER_AREA)
            # after every framerate frames
            if frame_id % math.floor(framerate) == 0:
                image_name = "image" + \
                    str(int(frame_id / math.floor(framerate)) + 1) + ".jpg"
                filename = os.path.join(directory, video_dir, image_name)
                print(filename)
                cv2.imwrite(filename, image)
        video.release()
        print('done')

    os.chdir(owd)


if __name__ == '__main__':
    source_directory = 'C:/Users/CTC219-PC01/Documents/thesis/bantupalli_xie/source_videos/'
    for gesture in os.listdir(source_directory):
        videos(os.path.join(source_directory, gesture))
