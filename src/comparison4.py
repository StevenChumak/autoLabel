import shutil
from cProfile import label

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utility import utils

shutil.rmtree("fig", ignore_errors=True)

raillabel_path = r"C:\Users\St\Desktop\Git\raillabel"

img = cv2.imread("001_input.png")  # read image
mask = cv2.imread("001_prediction.png")  # read image

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

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)

skeleton_rails = utils.pre_process(rail_img, open=False)
skeleton_trackbed = utils.pre_process(trackbed_img, open=True)

left, right, left_IImage, right_IImage = utils.rail_seperation(
    skeleton_rails, skeleton_trackbed
)
thicc_left, thicc_right, thicc_left_IImage, thicc_right_IImage = utils.rail_seperation(
    rail_img, skeleton_trackbed
)

approx_left, lin_left = utils.approximateKnots(
    left_IImage,
    nKnots=nKnots,
    minDistance=minDistance,
    interpolation=interpolation,
    log=logging,
    direction="left",
)
approx_right, lin_right = utils.approximateKnots(
    right_IImage,
    nKnots=nKnots,
    minDistance=minDistance,
    interpolation=interpolation,
    log=logging,
    direction="right",
)


# def creatRailCoordinates(left, right, approx_left, approx_right):
#     for point in approx_left:
#         print(point)


# def createRailPolygon(original, knots):
#     cv2.fillConvexPoly
#     pass


# left_rail_polygon = createRailPolygon(left, approx_left)

# railCoordinates = creatRailCoordinates(left, right, approx_left, approx_right)

# get bounding box for ego rails

x, y, w, h = cv2.boundingRect(rail_img)
margin = 5
############################################################################################################
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
ax = axes.ravel()

utils.plot_rail(ax[0], thicc_left, thicc_right, "*g")
ax[0].set_title("Original skeleton")
ax[0].legend()

utils.plot_rail(ax[1], approx_left, approx_right, ["*g", "*r"])
ax[1].set_title("Knotpoints")
ax[1].legend(["Approximated Knots Left", "Approximated Knots Right"])

utils.plot_ImagePoint(lin_left, plot=ax[2], color="*g")
utils.plot_ImagePoint(lin_right, plot=ax[2], color="*r")
ax[2].set_title("approx interpol")
ax[2].legend(["Rail Interpolation Left", "Rail Interpolation Right"])

utils.plot_ImagePoint(lin_left, plot=ax[3], color=":b", label="Interpolation")
utils.plot_ImagePoint(lin_right, plot=ax[3], color=":b", label="Interpolation")
utils.plot_rail(ax[3], left, right, ":g", label="Original")
ax[3].set_title("Comparison (points)")
ax[3].legend()


height = mask.shape[0]
width = mask.shape[1]
blank_image = np.zeros((height, width, 3), np.uint8)
for point in approx_left["image_points"]:
    circled = cv2.circle(
        blank_image, point.point, radius=3, color=(255, 255, 255), thickness=-1
    )
for point in approx_right["image_points"]:
    circled = cv2.circle(
        circled, point.point, radius=3, color=(255, 255, 255), thickness=-1
    )
added_image = cv2.addWeighted(img, 0.7, circled, 0.4, 0)
cropped = added_image[y - margin : y + h + margin, x - margin : x + w + margin]

ax[4].imshow(cropped)
ax[4].set_title("Overlayed Knotpoints")
ax[4].legend()

fig.tight_layout()
fig.savefig("figure.png")
fig.show()
############################################################################################################
blank_image = np.zeros((height, width), np.uint8)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(grayscale, 50, 150, L2gradient=True)
edges = np.array(np.where(edges != 0, 255, 0), dtype=np.uint8)

color_rail = np.array(np.stack([rail_img, rail_img, rail_img], axis=2), dtype=np.uint8)
color_edge = np.stack([edges, blank_image, blank_image], axis=2)

img_canny_overlay = cv2.addWeighted(img, 1, color_edge, 0.5, 0)
img_rail_overlay = cv2.addWeighted(img, 1, color_rail, 0.3, 0)
rail_canny_overlay = cv2.addWeighted(color_rail, 0.4, color_edge, 1, 0)

cropped_img_canny_overlay = img_canny_overlay[
    y - margin : y + h + margin, x - margin : x + w + margin
]
cropped_img_rail_overlay = img_rail_overlay[
    y - margin : y + h + margin, x - margin : x + w + margin
]
cropped_rail_canny_overlay = rail_canny_overlay[
    y - margin : y + h + margin, x - margin : x + w + margin
]

fig2, axes2 = plt.subplots(1, 3, figsize=(10, 10))
ax2 = axes2.ravel()

ax2[0].imshow(cropped_img_canny_overlay)
ax2[0].set_title("Img-Canny")

ax2[1].imshow(cropped_img_rail_overlay)
ax2[1].set_title("Img-Rail")

ax2[2].imshow(cropped_rail_canny_overlay)
ax2[2].set_title("Rails-Canny")

fig2.tight_layout()
fig2.savefig("figure2.png")
fig2.show()
############################################################################################################
input()
