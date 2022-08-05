import sys
import pathlib

import cv2
import matplotlib.pyplot as plt

from utility import utils

img = cv2.imread("../001_input.png")  # read image
mask = cv2.imread("../001_prediction.png")  # read image
img_copy = img.copy()

# colors used to color different classes
ego_color = [137, 49, 239]  # left_trackbed     | Blue-Violet
ego_rail_color = [242, 202, 25]  # left_rails        | Jonquil oder auch gelb
neighbor_color = [225, 24, 69]  # ego_trackbed      | Spanish Crimson

# convert colorized labal to class
# TODO: dont colorize the classes and let NN output the classes only?
trackbed_data = utils.mask_to_class(img, color=ego_color)
rail_data = utils.mask_to_class(img, color=ego_rail_color)

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)

skeleton_rails = utils.skel_close_dila(rail_img)
skeleton_trackbed = utils.skel_close_dila(trackbed_img)

left, right = utils.rail_seperation(skeleton_rails, skeleton_trackbed)

spline1 = utils.get_BSpline(left["x"], left["y"])
spline2 = utils.get_BSpline(right["x"], right["y"])

from scipy import interpolate
test = interpolate.UnivariateSpline(left["x"], left["y"])
knots = test.get_knots()
plt.plot(left["x"], left["y"], "g", lw=3)
plt.plot(right["x"], right["y"], "r", lw=3)

plt.plot(spline1["x"], spline1["y"], ":b", lw=3)
plt.plot(spline2["x"], spline2["y"], "--g", lw=3)

# lis = {"x": [], "y": []}
lis = []
for knot in knots:
    for i, x in enumerate(left["x"]):
        if x == knot:
            # lis["x"].append(knot)
            # lis["y"].append(left["y"][i])
            lis.append([knot, left["y"][i]])

    plt.axvline(knot,color='g')

# import numpy as np

# height = img.shape[0]
# width = img.shape[1]
# blank_image = np.zeros((height, width), np.uint8)

# print(np.array([left["xy"]], dtype=np.int32))

# cv2.fillConvexPoly(rail_img, np.array([left["xy"]], dtype=np.int32), color=(255, 255, 255))#, isClosed=True, thickness=1)
# plt.imshow(rail_img)

plt.show()
