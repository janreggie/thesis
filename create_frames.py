import os
import cv2
import glob
import Augmentor
#
#
def frames(dataset):
    # count = 0
    # frame = os.path.join(os.getcwd() + dataset)
    # vid = os.path.join(os.getcwd() + "\\videos\\Brady_15716.mov")
    # cap = cv2.VideoCapture(vid)
    # success, image = cap.read()
    # while success:
    #     crop_image = image[110:110+216, 0:0+270]
    #     cv2.imwrite(os.path.join(frame + '4_%d.jpg' % count), crop_image)
    #     success, image = cap.read()
    #     count += 1
    p = Augmentor.Pipeline("D:\\thesis\\test\\beer")
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

    p.sample(120)


if __name__ == '__main__':
    frames('\\training\\')

# os.chdir('D://thesis//videos//')
#
# path = 'D://thesis//again//'
#
# for file in glob.glob('*.jpg'):
#     image = cv2.imread(file)
#     crop_image = image[110:110+216, 0:0+270]
#     cv2.imwrite(os.path.join(path + file), crop_image)
