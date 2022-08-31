import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from skimage.morphology import (binary_closing, binary_dilation,
                                binary_erosion, binary_opening, closing,
                                dilation, erosion, opening, skeletonize)

import utility.image_splines
from utility.image_point import ImagePoint


def mask_to_class(mask, color, gray=False):
    """
    Change each value of a numpy array according to mapping.
    Returns a uint8 numpy array with changed values
    """

    if len(mask.shape) == 2:
        gray = True
    elif mask.shape[2] == 1:
        gray = True
    else:
        gray = False

    if gray:
        holder = np.zeros((mask.shape[0], mask.shape[1]))

        for i in color:
            holder += np.where(mask == i, 255, 0)
            # sanity check for debugging
            # test = np.unique(holder)
    else:
        r = color[0]
        g = color[1]
        b = color[2]
        # convert RGB color to BGR (opencv)
        holder = np.where(mask == [b, g, r], 255, 0)

    return np.uint8(holder)


def pre_process(
    img,
    dila=True,
    close=True,
    ero=False,
    open=False,
    skeleton=True,
    fast=True,
    mask=False,
):
    # reduce the objects to 1-2 pixel wide representation
    # dilatation is used first due to rails not being continues up until the end all of the time
    # more info at: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html

    if mask:
        mask = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
    else:
        mask = None

    if fast:
        img = img // 255
    if dila:
        if not fast:
            dilation(img, mask, out=img)
        else:
            binary_dilation(img, mask, out=img)
    if close:
        if not fast:
            closing(img, mask, out=img)
        else:
            binary_closing(img, mask, out=img)
    if ero:
        if not fast:
            erosion(img, mask, out=img)
        else:
            binary_erosion(img, mask, out=img)
    if open:
        if not fast:
            opening(img, mask, out=img)
        else:
            binary_opening(img, mask, out=img)
    if skeleton and fast:
        img = skeletonize(img)
    if fast:
        img = img * 255

    return np.array(img, dtype=np.uint8)


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


def plot_rail(ax: plt.Axes, left, right, color, label=None):
    if isinstance(color, list):
        ax.plot(left["x"], [-y for y in left["y"]], color[0])
        ax.plot(right["x"], [-y for y in right["y"]], color[1], label=label)
    else:
        ax.plot(left["x"], [-y for y in left["y"]], color)
        ax.plot(right["x"], [-y for y in right["y"]], color, label=label)

    return ax


def plot_ImagePoint(points, plot=None, color=None, label=None):
    if not color:
        color = ":g"
    x = []
    y = []
    for point in points:
        x.append(point.x)
        y.append(-point.y)
    if plot:
        plot.plot(x, y, color, label=label)
    else:
        plot = plt.axes()
        plot.plot(x, y, color, label=label)

    return plot


def plot_RailComp(
    interpol, original, previousResults, direction, dist=None, chosen=None
):
    plot = plot_ImagePoint(original, color=":b")
    plot = plot_ImagePoint(interpol, plot=plot, color=":g")

    if dist:
        x = []
        y = []
        for index in dist:
            x.append(original[index].x)
            y.append(-original[index].y)
        plot.plot(x, y, "*r")
        if chosen:
            x = []
            y = []
            for index in chosen:
                x.append(original[index].x)
                y.append(-original[index].y)
            plot.plot(x, y, "*k")
            plot.legend(
                ["Original", "Interpolation", "max Distances", "chosen Knotpoint"]
            )
        else:
            plot.legend(["Original", "Interpolation", "max Distances"])
    elif chosen:
        x = []
        y = []
        for index in chosen:
            x.append(original[index].x)
            y.append(-original[index].y)
        plot.plot(x, y, "*k")
        plot.legend(["Original", "Interpolation", "chosen Knotpoint"])
    else:
        plot.legend(["Original", "Interpolation"])
    if not os.path.exists("fig"):
        os.mkdir("fig")
    name = "fig/comparison_{} {}.png".format(direction, len(previousResults["ind"]) - 1)

    plot.figure.savefig(name)
    plt.close(plot.figure)


def calc_EuclideanArray(original, inteprolation):
    def euclidean(points):
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    original_array = np.vstack([point.point for point in original])
    interpol_array = np.vstack([point.point for point in inteprolation])

    stacked = np.column_stack((original_array, interpol_array))

    distance_list = []
    for points in stacked:
        distance_list.append(euclidean(points))

    distance_array = np.array(distance_list, dtype=np.float32)
    distance_sum = np.sum(distance_array).item()
    return distance_array, distance_sum


def get_CatMul(previousResults):
    steps = []
    for i in range(0, len(previousResults["ind"]) - 1):
        steps.append(previousResults["ind"][i + 1] - previousResults["ind"][i])

    return utility.image_splines.calculate_splines(
        tuple(previousResults["image_points"]), steps=tuple(steps)
    )


def linearInterpol(list, original):
    """
    I think this is a piecewise linear interpolation?
    """
    output = {"x": [], "y": [], "xy": [], "image_points": [], "m": [], "b": []}
    items = list["image_points"]
    for i in range(0, len(items) - 1):
        x1 = items[i].y
        x2 = items[i + 1].y
        y1 = items[i].x
        y2 = items[i + 1].x
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

        for i in range(list["ind"][i], list["ind"][i + 1]):
            x = original[i].y
            y = m * x + b
            if math.isnan(y):
                y = output["x"][-1]

            output["x"].append(int(y))
            output["y"].append(int(x))
            output["xy"].append([int(y), int(x)])
            output["image_points"].append(ImagePoint(int(y), int(x)))
            output["m"].append(m)
            output["b"].append(b)
    if len(original) != len(output["image_points"]):
        raise Exception(
            "Linear Interpolation: expected: {} points, calculated: {} points".format(
                len(original), len(output["image_points"])
            )
        )

    return output["image_points"]


def get_interpolation(interpolation, original, previousResults):
    if interpolation == "linear":
        interpol = linearInterpol(previousResults, original)
    elif interpolation == "catmull-rom":
        interpol = get_CatMul(previousResults)
    else:
        raise Exception(TypeError(f"{interpolation} is not supported"))
    return interpol


def approximateKnots(
    original,
    previousResults: dict = {},
    nKnots=20,
    minDistance=1,
    interpolation="catmull-rom",
    log=False,
    direction=None,
):
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
        return approximateKnots(
            original,
            previousResults=previousResults,
            minDistance=minDistance,
            interpolation=interpolation,
            log=log,
            direction=direction,
        )
    else:
        interpolated = get_interpolation(
            interpolation=interpolation,
            original=original,
            previousResults=previousResults,
        )

        # using the results from linear interpolation calculate the euclidean distance between intepolated coordinate and measurement
        original_array = np.vstack([point.point for point in original])
        interpolated_array = np.vstack([point.point for point in interpolated])

        if len(original_array) != len(interpolated_array):
            raise Exception(
                "Interpolated line does not have the same amount of points as original: {}:{}".format(
                    len(interpolated_array), len(original_array)
                )
            )

        dist = []
        for original_point, interpolated_point in zip(
            original_array, interpolated_array
        ):
            # calculate euclidean distance based on y-axis
            dist.append(
                round(np.linalg.norm((original_point - interpolated_point), 2), 2)
            )

        distance_array = np.array(dist, dtype=np.float32)
        distance_sum = np.sum(distance_array).item()
        previousResults["distance"].append(distance_sum)

        # get all indexes with max distance
        argmax_indexes = np.where(distance_array == np.amax(distance_array))[0].tolist()

        if np.amax(distance_array) > minDistance:
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
        else:
            if log:
                name = "fig/distance_{}.json".format(direction)
                with open(name, "w", encoding="utf-8") as file:
                    json.dump(
                        previousResults["distance"], file, ensure_ascii=False, indent=4
                    )
            return previousResults, interpolated

        if log:
            plot_RailComp(
                interpolated,
                original,
                previousResults,
                dist=argmax_indexes,
                chosen=argmax_index,
                direction=direction,
            )

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

        if len(previousResults["image_points"]) < nKnots:
            return approximateKnots(
                original,
                previousResults=previousResults,
                minDistance=minDistance,
                interpolation=interpolation,
                log=log,
                direction=direction,
            )
        else:
            if log:
                name = "fig/distance_{}.json".format(direction)
                with open(name, "w", encoding="utf-8") as file:
                    json.dump(
                        previousResults["distance"], file, ensure_ascii=False, indent=4
                    )

            return previousResults, interpolated


def approxRails(
    rail_img,
    labels={
        "trackbed": [1],
        "rails": [2],
    },
    interpolation="catmull-rom",
    nKnots=20,
    minDistance=4,
    logging=False,
):
    trackbed_data = mask_to_class(rail_img, color=labels["trackbed"])
    rail_data = mask_to_class(rail_img, color=labels["rails"])

    skeleton_trackbed = pre_process(trackbed_data)
    skeleton_rails = pre_process(rail_data)

    left, right, left_IImage, right_IImage = rail_seperation(
        skeleton_rails, skeleton_trackbed
    )

    approx_left, spline_left = approximateKnots(
        left_IImage,
        nKnots=nKnots,
        minDistance=minDistance,
        interpolation=interpolation,
        log=logging,
        direction="left",
    )
    approx_right, spline_right = approximateKnots(
        right_IImage,
        nKnots=nKnots,
        minDistance=minDistance,
        interpolation=interpolation,
        log=logging,
        direction="right",
    )

    return {
        "left": {"knots": approx_left["image_points"], "spline": spline_left},
        "right": {"knots": approx_right["image_points"], "spline": spline_right},
    }
