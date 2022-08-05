import shutil

import cv2
import matplotlib.pyplot as plt
import splines
import scipy.interpolate as interpolate

from utility import utils

shutil.rmtree("fig", ignore_errors=True)

raillabel_path = r"C:\Users\St\Desktop\Git\raillabel"

img = cv2.imread("001_input.png")  # read image
mask = cv2.imread("001_prediction.png")  # read image

nKnots = 20
logging= True

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

left, right, left_IImage, right_IImage = utils.rail_seperation(skeleton_rails, skeleton_trackbed)
thicc_left, thicc_right, thicc_left_IImage, thicc_right_IImage = utils.rail_seperation(rail_img, skeleton_trackbed)

approx_left, lin_left = utils.approximateKnots(left_IImage, nKnots=nKnots, log=logging)
approx_right, lin_right = utils.approximateKnots(right_IImage, nKnots=nKnots, log=logging)


def creatRailCoordinates(left, right, approx_left, approx_right):
    pass

def createRailPolygon(original, knots):
    cv2.fillConvexPoly
    pass


import numpy as np
def calculate_splines(
    points_arr,
    steps: int,
):
    """
    Calculate splines in between given Sequence of ImagePoints.
    :param points: Sequence of image points
    :param steps: Interpolation steps inbetween and including two points
    :return: List of interpolated ImagePoints
    """
    if len(points_arr) > 1:
        sp = splines.CatmullRom(points_arr, endconditions="natural")
        total_duration: int = sp.grid[-1] - sp.grid[0]
        t = np.linspace(0, total_duration, len(points_arr) * steps)
        splines_arr = sp.evaluate(t)

        spline_points = {"x": [], "y": [], "xy": []}
        for spline_arr in splines_arr:
            spline_points["x"].append(spline_arr[0].item())
            spline_points["y"].append(spline_arr[1].item())
            spline_points["xy"].append([spline_arr[0].item(), spline_arr[1].item()])
        return spline_points
    else:
        return []


left_rail_polygon = createRailPolygon(left, approx_left)

railCoordinates = creatRailCoordinates(left, right, approx_left, approx_right)

fig, axes = plt.subplots(2, 3, figsize=(10, 10))
ax = axes.ravel()

utils.plot_rail(ax[0], left, right, "*g")
ax[0].set_title("Original skeleton")
ax[0].legend()

utils.plot_rail(ax[1], approx_left, approx_right, ["*g", "*r"])
ax[1].set_title("approx")
ax[1].legend()

ax[2] = utils.plot_ImagePoint(lin_left, plot=ax[2], color="*g")
ax[2] = utils.plot_ImagePoint(lin_right, plot=ax[2], color="*r")
ax[2].set_title("approx interpol")
ax[2].legend()

ax[3] = utils.plot_ImagePoint(lin_left, plot=ax[3], color="b")
ax[3] = utils.plot_ImagePoint(lin_right, plot=ax[3], color="b")
utils.plot_rail(ax[3], left, right, "g")
ax[3].set_title("comparison")
ax[3].legend()

fig.tight_layout()
plt.show()