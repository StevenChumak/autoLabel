import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from skimage.morphology import binary_closing, binary_dilation, skeletonize

import utility.image_splines
from utility.image_point import ImagePoint


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
    points_left = []
    points_right = []

    for h in range(height - 1, 0, -1):
        height_counter = 0
        if np.any(image, where=(image[h, :] == 255)):
            line_pos = np.where(line[h, :])
            # in case rail are longer than trackbed
            # does this make sense?
            # TODO: maybe throw information for user to check?
            while not line_pos[0].any() > 0:
                height_counter += 1
                line_pos = np.where(line[h - height_counter, :])

            img_pos = np.where(image[h, :])

            for w in np.nditer(img_pos):
                x = w.item()
                y = h
                image_point = ImagePoint(
                    w.item(),  # x value
                    h,  # y value
                )
                height_counter = 0
                if w <= (np.asarray(line_pos).min() + np.asarray(line_pos).max()) // 2:
                    left["x"].append(x)
                    left["y"].append(y)
                    left["xy"].append([x, y])
                    points_left.append(image_point)
                else:
                    right["x"].append(x)
                    right["y"].append(y)
                    right["xy"].append([x, y])
                    points_right.append(image_point)

    return left, right, points_left, points_right


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


def plot_ImagePoint(points, plot=None, color=None):
    if not color:
        color = ":g"
    x = []
    y = []
    for point in points:
        x.append(point.x)
        y.append(-point.y)
    if plot:
        plot.plot(x, y, color)
    else:
        plot = plt.axes()
        plot.plot(x, y, color)
    return plot


def get_CatMul(previousResults):
    steps = []
    for i in range(0, len(previousResults["ind"]) - 1):
        steps.append(previousResults["ind"][i + 1] - previousResults["ind"][i])

    return utility.image_splines.calculate_splines(
        tuple(previousResults["image_points"]), steps=tuple(steps)
    )


def plot_RailComp(interpol, original, previousResults, dist=None):
    plot = plot_ImagePoint(original, color=":b")
    plot = plot_ImagePoint(interpol, plot=plot, color=":g")

    if dist:
        x = []
        y = []
        for index in dist:
            x.append(original[index].x)
            y.append(-original[index].y)
        plot.plot(x, y, "*r")
        plot.legend(["Original", "Interpolation", "max Distances"])
    else:
        plot.legend(["Original", "Interpolation"])
    if not os.path.exists("fig"):
        os.mkdir("fig")
    name = "fig/comparison_left {}.png".format(len(previousResults["ind"]) - 1)
    if os.path.exists(name):
        name = "fig/comparison_right {}.png".format(len(previousResults["ind"]) - 1)

    plot.figure.savefig(name)
    plt.close(plot.figure)


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


def calc_EuclideanArray(original, inteprolation):
    def euclidean(points):
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    original_array = np.vstack([point.point for point in original])
    interpol_array = np.vstack([point.point for point in inteprolation])

    # if len(original_array) == len(interpol_array)+1:
    #     inter = list(interpol_array)
    #     test = np.vstack([inter, inter[-1]])
    #     interpol_array = np.array(test)

    stacked = np.column_stack((original_array, interpol_array))

    distance_list = []
    for points in stacked:
        distance_list.append(euclidean(points))

    distance_array = np.array(distance_list, dtype=np.float32)
    distance_sum = np.sum(distance_array).item()
    return distance_array, distance_sum


def approximateKnot(original, previousResults: dict = {}, log=False):
    """
    Approximate knots
    """
    if not isinstance(original, (np.ndarray, np.generic, list, dict)):
        raise Exception("Expected list, dict or numpy array")

    if not previousResults:
        # instantiate previousResults with first and last point of original
        # middle = len(original)//2

        first_x = original[0].x
        # middle_x = original[middle].x
        last_x = original[-1].x

        first_y = original[0].y
        # middle_y = original[middle].y
        last_y = original[-1].y
        previousResults = {
            "x": [
                first_x,
                # middle_x,
                last_x,
            ],
            "y": [
                first_y,
                # middle_y,
                last_y,
            ],
            "xy": [
                [first_x, first_y],
                # [middle_x, middle_y],
                [last_x, last_y],
            ],
            "ind": [
                0,
                # middle,
                len(original),
            ],
            "image_points": [
                ImagePoint(first_x, first_y),
                # ImagePoint(middle_x, middle_y),
                ImagePoint(last_x, last_y),
            ],
            "distance": [],
        }

        # distance measurements depend on interpolation,
        # if different interpolation is required changed it here
        # interpol = linearInterpol(previousResults, original)
        interpol = get_CatMul(previousResults)
    else:
        # interpol = linearInterpol(previousResults, original)
        interpol = get_CatMul(previousResults)

        # using the results from linear interpolation calculate the euclidean distance between intepolated coordinate and measurement
        original_arr = np.vstack([point.point for point in original])
        interpol_arr = np.vstack([point.point for point in interpol])

        if len(original_arr) != len(interpol_arr):
            raise Exception(
                "Interpolated line does not have the same amount of points as original: {}:{}".format(
                    len(interpol_arr), len(original_arr)
                )
            )

        dist = []
        for orig_point, inter_point in zip(original_arr, interpol_arr):
            # calculate euclidean distance based on y-axis
            dist.append(round(np.linalg.norm((orig_point - inter_point), 2), 2))

        distance_array = np.array(dist, dtype=np.float32)
        distance_sum = np.sum(distance_array).item()
        previousResults["distance"].append(distance_sum)

        # get all indexes with max distance
        argmax_indexes = np.where(distance_array == np.amax(distance_array))[0].tolist()
        if log:
            plot_RailComp(interpol, original, previousResults, dist=argmax_indexes)

        #  delete indexes which already have a knot in previousResults
        for elem in previousResults["ind"]:
            argmax_index = [x for x in argmax_indexes if x != elem]

        if len(argmax_index) == 1:
            argmax_index = [argmax_index[0]]
        else:
            # take the middle knot
            middle = len(argmax_index) // 2
            try:
                argmax_index = [argmax_index[middle]]
                # argmax_index = [argmax_index[0], argmax_index[-1]]
            except:
                raise Exception(
                    "Distance Index outta bounds {}".format(argmax_index[middle])
                )
                # raise Exception("Distance Index outta bounds {} and {}".format(argmax_index[0], argmax_index[-1]))

        # use coordinates of point with biggest distance as knot
        for i in argmax_index:
            new_x = original[i].x
            new_y = original[i].y

            previousResults["x"].append(new_x)
            previousResults["y"].append(new_y)
            previousResults["xy"].append([new_x, new_y])
            previousResults["image_points"].append(ImagePoint(new_x, new_y))
            previousResults["ind"].append(i)

        # sort lists
        sorted_index = np.array(previousResults["ind"]).argsort()
        previousResults["x"] = (np.array(previousResults["x"])[sorted_index]).tolist()
        previousResults["y"] = (np.array(previousResults["y"])[sorted_index]).tolist()
        previousResults["xy"] = (np.array(previousResults["xy"])[sorted_index]).tolist()
        previousResults["image_points"] = (
            np.array(previousResults["image_points"])[sorted_index]
        ).tolist()
        previousResults["ind"] = (
            np.array(previousResults["ind"])[sorted_index]
        ).tolist()

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


def approximateKnots(points, nKnots=20, log=False):
    approximatedKnots, linearInterpolation = approximateKnot(points, log=log)
    while len(approximatedKnots["x"]) < nKnots:
        approximatedKnots, linearInterpolation = approximateKnot(
            points, previousResults=approximatedKnots, log=log
        )

    if log:
        name = "fig/distance_left.json"
        if os.path.exists(name):
            name = "fig/distance_right.json"
        with open(name, "w", encoding="utf-8") as file:
            json.dump(approximatedKnots["distance"], file, ensure_ascii=False, indent=4)

    approximatedKnots = removeDuplicates(approximatedKnots)

    return approximatedKnots, linearInterpolation
