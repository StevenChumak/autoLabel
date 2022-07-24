import numpy as np
from skimage.morphology import skeletonize, binary_dilation, binary_closing
from scipy import interpolate


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

    left = {"x": [], "y": []}
    right = {"x": [], "y": []}

    for h in range(0, height, 1):
        if np.any(image, where=(image[h, :] == 255)):
            img_pos = np.where(image[h, :])
            line_pos = np.where(line[h, :])
            for w in np.nditer(img_pos):
                if w <= (np.asarray(line_pos).min() + np.asarray(line_pos).max()) // 2:
                    # left.append([w.item(), h])
                    left["x"].append(w.item())
                    left["y"].append(h)
                else:
                    # right.append([w.item(), h])
                    right["x"].append(w.item())
                    right["y"].append(h)

    return left, right


def get_spline(x, y):

    tck, u = interpolate.splprep([x, y], k=4, s=25)
    spline = interpolate.splev(u, tck)
    return {"x": spline[0], "y": spline[1]}
