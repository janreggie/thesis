import os
import cv2
import glob
#
#
# def frames(dataset):
#     count = 0
#     frame = os.path.join(os.getcwd() + dataset)
#     vid = os.path.join(os.getcwd() + "\\videos\\again\\Brady_4707.mov")
#     cap = cv2.VideoCapture(vid)
#     success, image = cap.read()
#     while success:
#         crop_image = image[110:110+216, 0:0+270]
#         cv2.imwrite(os.path.join(frame + '1_%d.jpg' % count), crop_image)
#         success, image = cap.read()
#         count += 1
#
#
# if __name__ == '__main__':
#     frames('\\videos\\again\\')

os.chdir('C://Users//kbantupa//Desktop//new//videos//again//')

path = 'C://Users//kbantupa//Desktop//new//videos//again//'

for file in glob.glob('*.jpg'):
    image = cv2.imread(file)
    crop_image = image[110:110+216, 0:0+270]
    cv2.imwrite(os.path.join(path + file), crop_image)
