import shutil
from cProfile import label

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utility import utils

shutil.rmtree("fig", ignore_errors=True)

raillabel_path = r"C:\Users\St\Desktop\Git\raillabel"

img_path = "001_input.png"
mask_path = "001_prediction.png"
test_path = "bad_pred.png"
img = cv2.imread(img_path)  # read image
mask = cv2.imread(mask_path)  # read mask
test = cv2.imread(test_path)  # read image

nKnots = 20
logging = False
minDistance = 4
interpolation = "catmull-rom"
# interpolation = "linear"

# colors used to color different classes
ego_color = [137, 49, 239]  # left_trackbed     | Blue-Violet
ego_rail_color = [242, 202, 25]  # left_rails        | Jonquil oder auch gelb
neighbor_color = [225, 24, 69]  # ego_trackbed      | Spanish Crimson

# convert colorized labal to class
# TODO: dont colorize the classes and let NN output the classes only?
trackbed_data = utils.mask_to_class(mask, color=ego_color)
rail_data = utils.mask_to_class(mask, color=ego_rail_color)
neighor_data = utils.mask_to_class(mask, color=neighbor_color)

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)
neighor_img = cv2.cvtColor(neighor_data, cv2.COLOR_BGR2GRAY)
gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

skeleton_rails = utils.pre_process(rail_img, ero=False, open=False)
skeleton_trackbed = utils.pre_process(trackbed_img, open=True)
neighor_trackbed = utils.pre_process(neighor_img, open=True)
test_processed = utils.pre_process(gray_test, open=False, skeleton=False, fast=False)


def connected_component_label(path=None, img=None):
    if path:
        # Getting the input image
        img = cv2.imread(path, 0)
        # Converting those pixels with values 1-127 to 0 and others to 1
        img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]

    # Applying cv2.connectedComponents()
    num_labels, labels = cv2.connectedComponents(img)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    # Showing Original Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Orginal Image")
    plt.show()

    # Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Image after Component Labeling")
    plt.show()


connected_component_label(img=neighor_trackbed)
