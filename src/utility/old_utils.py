import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.spatial import distance
from skimage.morphology import (binary_closing, binary_dilation,
                                binary_erosion, skeletonize)


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
        skeletonize(binary_erosion(binary_closing(binary_dilation(img // 255)))) * 255,
        dtype=np.uint8,
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
        interpol = linearInterpol(previousResults, original)
        # N = len(original["x"])
        # xmin, xmax = (
        #     np.array(previousResults["x"]).min(),
        #     np.array(previousResults["x"]).max(),
        # )
        # xx = np.linspace(xmin, xmax, N)
        # t, c, k = interpolate.splrep(
        #     np.array(previousResults["x"]), np.array(previousResults["y"]), s=0, k=2
        # )
        # spline = interpolate.BSpline(t, c, k, extrapolate=False)
        # interpol = {
        #     "x": xx,
        #     "y": spline(xx),
        #     "xy": np.column_stack(
        #         (np.array(xx, dtype=int), np.array(spline(xx), dtype=int))
        #     ),
        # }
    else:
        interpol = linearInterpol(previousResults, original)
        # k = len(previousResults["x"]) - 1 if len(previousResults["x"]) < 5 else 4
        # N = len(original["x"])
        # xmin, xmax = (
        #     np.array(previousResults["x"]).min(),
        #     np.array(previousResults["x"]).max(),
        # )
        # xx = np.linspace(xmin, xmax, N)
        # t, c, k = interpolate.splrep(
        #     previousResults["x"], previousResults["y"], s=0, k=k
        # )
        # spline = interpolate.BSpline(t, c, k, extrapolate=False)
        # interpol = {
        #     "x": xx,
        #     "y": spline(xx),
        #     "xy": np.column_stack(
        #         (np.array(xx, dtype=int), np.array(spline(xx), dtype=int))
        #     ),
        # }
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
