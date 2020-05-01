import Augmentor
#
#


def frames(dataset, output=""):
    '''
    Opens images in dataset and augments them, writing to output folder.
    Default writeout is subfolder named "output" in dataset
    '''
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
    augment = Augmentor.Pipeline(dataset, output)
    augment.rotate(probability=0.7, max_left_rotation=10,
                   max_right_rotation=10)
    augment.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

    augment.sample(120)


if __name__ == '__main__':
    frames("C:/Users/ryZen/Downloads/Thesis/source_videos/validate/beer/video_4",
           "C:/Users/ryZen/Downloads/Thesis/source_videos/validate/beer/video_4")
