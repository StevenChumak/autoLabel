import cv2
import matplotlib.pyplot as plt

from utility import old_utils as utils

raillabel_path = r"C:\Users\St\Desktop\Git\raillabel"

img = cv2.imread("001_input.png")  # read image
mask = cv2.imread("001_prediction.png")  # read image
img_copy = mask.copy()

# colors used to color different classes
ego_color = [137, 49, 239]  # left_trackbed     | Blue-Violet
ego_rail_color = [242, 202, 25]  # left_rails        | Jonquil oder auch gelb
neighbor_color = [225, 24, 69]  # ego_trackbed      | Spanish Crimson

# convert colorized labal to class
# TODO: dont colorize the classes and let NN output the classes only?
trackbed_data = utils.mask_to_class(mask, color=ego_color)
rail_data = utils.mask_to_class(mask, color=ego_rail_color)

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)

skeleton_rails = utils.skel_close_dila(rail_img)
skeleton_trackbed = utils.skel_close_dila(trackbed_img)

left, right = utils.rail_seperation(skeleton_rails, skeleton_trackbed)
thicc_left, thicc_right = utils.rail_seperation(rail_img, skeleton_trackbed)


spline1 = utils.get_BSpline(left["x"], left["y"])
spline2 = utils.get_BSpline(right["x"], right["y"])

import numpy as np

height = mask.shape[0]
width = mask.shape[1]

fig, axes = plt.subplots(2, 3, figsize=(10, 10))
ax = axes.ravel()

utils.plot_rail(ax[0], left, right, "g")
ax[0].set_title("Original skeleton")

utils.plot_rail(ax[1], spline1, spline2, [":r", ":g"])
ax[1].set_title("Spline")

utils.plot_rail(ax[2], left, right, "g")
utils.plot_rail(ax[2], spline1, spline2, ":r")
ax[2].set_title("Overlay")

# Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
x, y, w, h = cv2.boundingRect(rail_img)
margin = 5

blank_image = np.zeros((height, width, 3), np.uint8)
for point in left["xy"]:
    # for point in thicc_left["xy"]:
    blank_image = cv2.circle(
        blank_image, point, radius=0, color=(255, 255, 255), thickness=-1
    )

cv2.imwrite("skeleton_rails.png", blank_image)


# for point in thicc_right["xy"]:
#     blank_image = cv2.circle(blank_image, point, radius=0, color=(255,255,255), thickness=-1)

# added_image = cv2.addWeighted(img, 0.7, blank_image, 0.4, 0)
# utils.plot_rail(ax[3], thicc_left, thicc_right, ["g", "r"])
# ax[3].imshow(added_image[y-margin:y+h+margin, x-margin:x+w+margin])
# ax[3].set_title("Original")

################################################################################################
from scipy.interpolate import UnivariateSpline

# right_x_inv = np.array(right["y"])
# right_y_inv = np.array(right["x"])
# right_x_inds = right_x_inv.argsort()

# right_x_inv = right_x_inv[right_x_inds]
# right_y_inv = right_y_inv[right_x_inds]

# left_x_inv = np.array(left["y"])
# left_y_inv = np.array(left["x"])
# left_x_inds = left_x_inv.argsort()

# left_x_inv = left_x_inv[left_x_inds]
# left_y_inv = left_y_inv[left_x_inds]

# test = UnivariateSpline(left_x_inv, left_y_inv, k=4)
# knots = test.get_knots()
# print(f"UnivariateSpline: {len(knots)}")

# test2 = UnivariateSpline(right_x_inv, right_y_inv, k=4)
# knots2 = test2.get_knots()
# print(f"UnivariateSpline: {len(knots2)}")

# lis = {"x": [], "y": []}
# for knot in knots:
#     for i, y in enumerate(left["y"]):
#         if y == knot:
#             lis["x"].append(left["x"][i])
#             lis["y"].append(y)

# ax[4].plot(left_y_inv, [-y for y in left_x_inv], ":r")
# ax[4].plot(lis["x"], [-y for y in lis["y"]], "g*")

# lis2 = {"x": [], "y": []}
# for knot in knots2:
#     for i, y in enumerate(right["y"]):
#         if y == knot:
#             lis2["x"].append(right["x"][i])
#             lis2["y"].append(y)

# ax[4].plot(right_y_inv, [-y for y in right_x_inv], ":g")
# ax[4].plot(lis2["x"], [-y for y in lis2["y"]], "r*")

# test_right = np.rot90(right["xy"], k=1)
# test_left = np.rot90(left["xy"], k=1)

# test_left = {"x": test_left[0], "y":test_left[1]}
# test_right = {"x": test_right[0], "y":test_right[1]}

# test_right = np.rot90([test_right["x"], test_right["y"]], k=-1)
# test_left = np.rot90([test_left["x"], test_left["y"]], k=-1)

# test_left = {"x": test_left[0], "y":test_left[1]}
# test_right = {"x": test_right[0], "y":test_right[1]}

# utils.plot_rail(ax[4], test_left, test_right, [":r",":g"])

# yindex = test_right[0].argsort()
# x = test_right
#
# test2 = UnivariateSpline(test_right["x"], test_right[""], k=4)
# ax[4].plot(test2["x"], test2["y"], ":r")


# blank_image = np.zeros((height, width, 3), np.uint8)
# for point in left["xy"]:
#     blank_image = cv2.circle(blank_image, point, radius=0, color=(255,255,255), thickness=-1)
# for point in right["xy"]:
#     blank_image = cv2.circle(blank_image, point, radius=0, color=(255,255,255), thickness=-1)
# added_image = cv2.addWeighted(img, 0.7, blank_image, 0.4, 0)
# ax[4].imshow(added_image[y-margin:y+h+margin, x-margin:x+w+margin])

# blank_image = np.zeros((height, width, 3), np.uint8)
# imagecontours, _ = cv2.findContours(rail_img//255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for count in imagecontours:
#     epsilon = 0.1 * cv2.arcLength(count, True)
#     approximations = cv2.approxPolyDP(count, epsilon, True)
#     cv2.drawContours(blank_image, [approximations], 0, (255, 255, 255), 3)

# added_image = cv2.addWeighted(img, 0.7, blank_image, 0.4, 0)
# added_image = added_image[y-margin:y+h+margin, x-margin:x+w+margin]
# ax[4].imshow(added_image, cmap='gray', vmin=0, vmax=255)

################################################################################################

# coppy = img[y-margin:y+h+margin, x-margin:x+w+margin]
# added_image = cv2.addWeighted(img, 0.7, blank_image, 0.4, 0)
# ax[5].imshow(coppy)

# import splines
# import numpy as np

# points_arr = np.vstack([point for point in left["xy"]])

# sp = splines.CatmullRom(points_arr, endconditions="natural")
# total_duration: int = sp.grid[-1] - sp.grid[0]
# t = np.linspace(0, total_duration, len(points_arr) * 15)
# splines_arr = sp.evaluate(t)
# spline_points = []
# for spline_arr in splines_arr:
#     spline_point = [
#         spline_arr[0].item(),
#         spline_arr[1].item(),
#     ]
#     spline_points.append(spline_point)
# ax[4].plot(spline_points[0], spline_points[1])

# test2 = interpolate.UnivariateSpline(spline1["x"], spline1["y"], k=4)
# knots2 = test2.get_knots()
# print(len(knots2))

# lis = {"x": [], "y": []}
# for knot in knots2:
#     for i, x in enumerate(left["x"]):
#         if x == knot:
#             lis["x"].append(knot)
#             lis["y"].append(left["y"][i])

#     ax[1].axvline(knot, color='b')


fig.tight_layout()
plt.show()
