import shutil

import cv2
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import splines

from utility import old_utils

shutil.rmtree("fig", ignore_errors=True)


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
trackbed_data = old_utils.mask_to_class(mask, color=ego_color)
rail_data = old_utils.mask_to_class(mask, color=ego_rail_color)

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)

skeleton_rails = old_utils.skel_close_dila(rail_img)
skeleton_trackbed = old_utils.skel_close_dila(trackbed_img)

left, right = old_utils.rail_seperation(skeleton_rails, skeleton_trackbed)
thicc_left, thicc_right = old_utils.rail_seperation(rail_img, skeleton_trackbed)

nKnots = 20
minDistance = 2

approx_left, lin_left = old_utils.approximateKnots(
    left, nKnots=nKnots, minDistance=minDistance
)
approx_right, lin_right = old_utils.approximateKnots(
    right, nKnots=nKnots, minDistance=minDistance
)


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

old_utils.plot_rail(ax[0], left, right, "*g")
ax[0].set_title("Original skeleton")

old_utils.plot_rail(ax[1], approx_left, approx_right, ["*g", "*r"])
ax[1].set_title("approx")

old_utils.plot_rail(ax[2], lin_left, lin_right, ["*g", "*r"])
ax[2].set_title("approx interpol")

print(np.argmin(lin_left["x"]))

s1 = calculate_splines([approx_left["x"], approx_left["y"]], 15)
s2 = calculate_splines([approx_right["x"], approx_right["y"]], 15)

old_utils.plot_rail(ax[3], s1, s2, ["*g", "*r"])
ax[3].set_title("catmull romspline wth knots")

left_spline = old_utils.get_BSpline(left["x"], left["y"], knots=approx_left["xy"])
right_spline = old_utils.get_BSpline(right["x"], right["y"], knots=approx_right["xy"])

old_utils.plot_rail(ax[4], left_spline, right_spline, ["*g", "*r"])
ax[4].set_title("b spline interpolation with knots")

# N = 100
# xmin, xmax = approx_left["x"].min(), approx_left["x"].max()
# xx = np.linspace(xmin, xmax, N)
# t, c, k = interpolate.splrep(approx_left["x"], approx_left["y"], s=0, k=4)
# spline = interpolate.BSpline(t, c, k, extrapolate=False)
# plt.plot(xx, spline(xx), 'g', label='BSpline')
# ax[5].plot()

# xmin, xmax = approx_right["x"].min(), approx_right["x"].max()
# xx = np.linspace(xmin, xmax, N)
# t, c, k = interpolate.splrep(approx_right["x"], approx_right["y"], s=0, k=4)
# spline = interpolate.BSpline(t, c, k, extrapolate=False)
# plt.plot(xx, spline(xx), 'r', label='BSpline')
# ax[5].plot()

# ax[5].set_title("b spline interpolation with knots 2")


fig.tight_layout()
plt.show()
