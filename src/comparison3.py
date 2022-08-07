import shutil

import cv2
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import splines

from utility import utils

shutil.rmtree("fig", ignore_errors=True)

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.spatial import distance
from skimage.morphology import binary_closing, binary_dilation, skeletonize


def mask_to_class(mask, color):
    """
    Change each value of a numpy array according to mapping.
    Returns a uint8 numpy array with changed values
    """

    r = color[0]
    g = color[1]
    b = color[2]

    holder = np.where(mask == [b, g, r], 255, 0)

    return np.uint8(holder)


def skel_close_dila(img):
    # reduce the objects to 1-2 pixel wide representation
    # dilatation is used first due to rails not being continues up until the end all of the time
    # more info at: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html
    return np.array(
        skeletonize(binary_closing(binary_dilation(img // 255))) * 255, dtype=np.uint8
    )


def line_compare(image, line):
    """
    Parameters:
        image: numpy array of a grayscale image
        line: numpy array of a grazscale image depicting the seperation line

    Return:
        numpy array with the shape of image where pixels to the left of the seperation line have a value of 255 and pixel to the right of the seperation line have a value of 123

    """

    height = image.shape[0]
    width = image.shape[1]
    # channels = img.shape[2]

    blank_image = np.zeros((height, width), np.uint8)

    assert (
        height == line.shape[0] and width == line.shape[1]
    ), "Dimension mismatch: got {},{} expected {},{}".format(
        line.shape[0], line.shape[1], height, width
    )

    for h in range(0, height):
        if np.any(image, where=(image[h, :] == 255)):
            img_pos = np.where(image[h, :])
            line_pos = np.where(line[h, :])
            for w in np.nditer(img_pos):
                blank_image[h, w] = 255 if w <= (np.asarray(line_pos).min()) else 123

    return blank_image


def rail_seperation(image, line):
    """
    seperate a pixel in image into left and right depending on whether a pixel is to the left of "line" or to the right

    Parameters:
        image: numpy array of a grayscale image
        line: numpy array of a grayscale image depicting the seperation line

    Return:
        2 dicts of x, y corrdinates for left and right rail

    """

    height = image.shape[0]
    width = image.shape[1]
    # channels = img.shape[2]

    assert (
        height == line.shape[0] and width == line.shape[1]
    ), "Dimension mismatch: got {},{} expected {},{}".format(
        line.shape[0], line.shape[1], height, width
    )

    left = {"x": [], "y": [], "xy": []}
    right = {"x": [], "y": [], "xy": []}

    for h in range(height - 1, 0, -1):
        height_counter = 0
        if np.any(image, where=(image[h, :] == 255)):
            img_pos = np.where(image[h, :])
            line_pos = np.where(line[h, :])
            while not line_pos[0].any() > 0:
                height_counter += 1
                line_pos = np.where(line[h - height_counter, :])

            for w in np.nditer(img_pos):
                height_counter = 0
                if w <= (np.asarray(line_pos).min() + np.asarray(line_pos).max()) // 2:
                    left["x"].append(w.item())
                    left["y"].append(h)
                    left["xy"].append([w.item(), h])
                else:
                    right["x"].append(w.item())
                    right["y"].append(h)
                    right["xy"].append([w.item(), h])

    return left, right


def get_BSpline(x, y, k=4, s=0, knots=None):
    tck, u = interpolate.splprep([x, y], k=k, s=s, t=knots)
    spline = interpolate.splev(u, tck)
    xy = np.column_stack(
        (np.array(spline[0], dtype=int), np.array(spline[1], dtype=int))
    )
    return {"x": spline[0], "y": spline[1], "xy": xy, "tck": tck, "u": u}


def plot_rail(ax: plt.Axes, left, right, color):
    if isinstance(color, list):
        ax.plot(left["x"], [-y for y in left["y"]], color[0])
        ax.plot(right["x"], [-y for y in right["y"]], color[1])
    else:
        ax.plot(left["x"], [-y for y in left["y"]], color)
        ax.plot(right["x"], [-y for y in right["y"]], color)

    return ax


import os

import matplotlib.pyplot as plt


def linearInterpol(list, points, compare=True):
    """
    I think this is a piecewise linear interpolation?
    """
    output = {"x": [], "y": [], "xy": [], "m": [], "b": []}
    for i in range(0, len(list["x"]) - 1):
        x1 = list["y"][i]
        x2 = list["y"][i + 1]
        y1 = list["x"][i]
        y2 = list["x"][i + 1]
        try:
            m = (y1 - y2) / (x1 - x2)
        except:
            try:
                m = output["m"][-1]
            except:
                m = 1
        try:
            b = (x1 * y2 - x2 * y1) / (x1 - x2)
        except:
            try:
                b = output["b"][-1]
            except:
                b = 0

        for x in points["y"][list["ind"][i] : list["ind"][i + 1]]:
            y = m * x + b
            if math.isnan(y):
                y = output["x"][-1]

            output["x"].append(int(y))
            output["y"].append(int(x))
            output["xy"].append([int(y), int(x)])
            output["m"].append(m)
            output["b"].append(b)
    if len(points["y"]) != len(output["x"]):
        print(len(points))
        print(len(output["x"]))
        print("expected: {}, got: {}".format(len(points), len(output["x"])))
    if compare:
        plt.plot(points["x"], [-y for y in points["y"]], ":g")
        plt.plot(output["x"], [-y for y in output["y"]], ":b")
        plt.legend(["Original", "Interpolation"])
        if not os.path.exists("fig"):
            os.mkdir("fig")
        name = "fig/comparison_left {}.png".format(len(list["ind"]) - 1)
        if os.path.exists(name):
            name = "fig/comparison_right {}.png".format(len(list["ind"]) - 1)

        plt.savefig(name)
        plt.close()

    return output


def approximateKnot(original, previousResults: dict = {}, minDistance=0):
    """
    Approximate knots
    """
    if not isinstance(original, (np.ndarray, np.generic, list, dict)):
        raise Exception("Expected list, dict or numpy array")

    if not previousResults:
        # instantiate previousResults with first and last point of original
        middle = len(original["x"]) // 2
        previousResults = {
            "x": [original["x"][0], original["x"][middle], original["x"][-1]],
            "y": [original["y"][0], original["y"][middle], original["y"][-1]],
            "xy": [
                [original["x"][0], original["y"][0]],
                [original["x"][middle], original["y"][middle]],
                [original["x"][-1], original["y"][-1]],
            ],
            "ind": [0, middle, len(original["xy"])],
        }

        # distance measurements depend on interpolation,
        # if different interpolation is required changed it here
        # interpol = linearInterpol(previousResults, original)
        N = len(original["x"])
        xmin, xmax = (
            np.array(previousResults["x"]).min(),
            np.array(previousResults["x"]).max(),
        )
        xx = np.linspace(xmin, xmax, N)
        t, c, k = interpolate.splrep(
            np.array(previousResults["x"]), np.array(previousResults["y"]), s=0, k=2
        )
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        interpol = {
            "x": xx,
            "y": spline(xx),
            "xy": np.column_stack(
                (np.array(xx, dtype=int), np.array(spline(xx), dtype=int))
            ),
        }
    else:
        # interpol = linearInterpol(previousResults, original)
        k = len(previousResults["x"]) - 1 if len(previousResults["x"]) < 5 else 4
        N = len(original["x"])
        xmin, xmax = (
            np.array(previousResults["x"]).min(),
            np.array(previousResults["x"]).max(),
        )
        xx = np.linspace(xmin, xmax, N)
        t, c, k = interpolate.splrep(
            previousResults["x"], previousResults["y"], s=0, k=k
        )
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        interpol = {
            "x": xx,
            "y": spline(xx),
            "xy": np.column_stack(
                (np.array(xx, dtype=int), np.array(spline(xx), dtype=int))
            ),
        }
        dist = []

        # using the results from linear interpolation calculate the euclidean distance between intepolated coordinate and measurement
        for i in range(0, len(original["xy"])):
            dist.append(distance.euclidean(original["xy"][i], interpol["xy"][i]))

        # get index of biggest distance
        dist_array = np.array(dist, dtype=np.int32)

        if np.argmax(dist_array) > minDistance:
            # get all indexes with max distance
            argmax_index = np.where(dist_array == np.amax(dist_array))
            argmax_index = np.array(argmax_index).flatten()
            # delete indexes which already have a knot in previousResults
            for elem in previousResults["ind"]:
                argmax_index = argmax_index[argmax_index != elem]

            if len(argmax_index) == 1:
                argmax_index = argmax_index.item()
            else:
                # take the middle knot
                middle = len(argmax_index) // 2
                try:
                    argmax_index = argmax_index.item(middle)
                except:
                    print(middle)
        else:
            argmax_index = 0

        # use coordinates of point with biggest distance as knot
        new_x = original["x"][argmax_index]
        new_y = original["y"][argmax_index]

        previousResults["x"].append(new_x)
        previousResults["y"].append(new_y)
        previousResults["xy"].append([new_x, new_y])
        previousResults["ind"].append(argmax_index)

        # sort lists
        sorted_index = np.array(previousResults["ind"]).argsort()
        previousResults["x"] = (np.array(previousResults["x"])[sorted_index]).tolist()
        previousResults["y"] = (np.array(previousResults["y"])[sorted_index]).tolist()
        previousResults["xy"] = (np.array(previousResults["xy"])[sorted_index]).tolist()
        previousResults["ind"] = (
            np.array(previousResults["ind"])[sorted_index]
        ).tolist()

        plt.plot(original["x"], [-y for y in original["y"]], ":g")
        plt.plot(interpol["x"], [-y for y in interpol["y"]], ":b")
        plt.legend(["Original", "Interpolation"])
        if not os.path.exists("fig"):
            os.mkdir("fig")
        name = "fig/comparison_left {}.png".format(len(previousResults["ind"]) - 1)
        if os.path.exists(name):
            name = "fig/comparison_right {}.png".format(len(previousResults["ind"]) - 1)

        plt.savefig(name)
        plt.close()

    return previousResults, interpol


def removeDuplicates(target):
    # use indexes gives by target["ind"] to determin duplicates and remove these duplices
    uniques_list = np.unique(target["ind"])
    for i in range(0, len(uniques_list)):
        listed = np.asarray(np.where(target["ind"] == uniques_list[i])).flatten()
        if listed.shape[0] > 2:
            target["x"] = np.delete(target["x"], listed[0:-1])
            target["y"] = np.delete(target["y"], listed[0:-1])
            target["ind"] = np.delete(target["ind"], listed[0:-1])

    return target


def approximateKnots(points, nKnots=20, minDistance=1):
    approximatedKnots, linearInterpolation = approximateKnot(
        points, minDistance=minDistance
    )
    while len(approximatedKnots["x"]) < nKnots:
        approximatedKnots, linearInterpolation = approximateKnot(
            points, previousResults=approximatedKnots, minDistance=minDistance
        )

    approximatedKnots = removeDuplicates(approximatedKnots)

    return approximatedKnots, linearInterpolation


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
trackbed_data = mask_to_class(mask, color=ego_color)
rail_data = mask_to_class(mask, color=ego_rail_color)

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)

skeleton_rails = skel_close_dila(rail_img)
skeleton_trackbed = skel_close_dila(trackbed_img)

left, right = rail_seperation(skeleton_rails, skeleton_trackbed)
thicc_left, thicc_right = rail_seperation(rail_img, skeleton_trackbed)

nKnots = 20
minDistance = 2

approx_left, lin_left = approximateKnots(left, nKnots=nKnots, minDistance=minDistance)
approx_right, lin_right = approximateKnots(
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

plot_rail(ax[0], left, right, "*g")
ax[0].set_title("Original skeleton")

plot_rail(ax[1], approx_left, approx_right, ["*g", "*r"])
ax[1].set_title("approx")

plot_rail(ax[2], lin_left, lin_right, ["*g", "*r"])
ax[2].set_title("approx interpol")

print(np.argmin(lin_left["x"]))

s1 = calculate_splines([approx_left["x"], approx_left["y"]], 15)
s2 = calculate_splines([approx_right["x"], approx_right["y"]], 15)

plot_rail(ax[3], s1, s2, ["*g", "*r"])
ax[3].set_title("catmull romspline wth knots")

left_spline = get_BSpline(left["x"], left["y"], knots=approx_left["xy"])
right_spline = get_BSpline(right["x"], right["y"], knots=approx_right["xy"])

plot_rail(ax[4], left_spline, right_spline, ["*g", "*r"])
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
