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

skeleton_rails = utils.pre_process(rail_img)
skeleton_trackbed = utils.pre_process(trackbed_img)

left, right, left_IImage, right_IImage = utils.rail_seperation(
    skeleton_rails, skeleton_trackbed
)
thicc_left, thicc_right, thicc_left_IImage, thicc_right_IImage = utils.rail_seperation(
    rail_img, skeleton_trackbed
)

x, y, w, h = cv2.boundingRect(rail_img)
margin = 50


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors="r")
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


# img = img[y - margin : y + h + margin, x - margin : x + w + margin]
height = img.shape[0]
width = img.shape[1]

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

import logging

import morphsnakes as ms

logging.basicConfig(level=logging.DEBUG)

gimg = ms.inverse_gaussian_gradient(grayscale)
callback = visual_callback_2d(img)

init_ls = np.zeros((height, width), np.uint8)
init_ls[y : y + h, x : x + w] = 1

ms.morphological_geodesic_active_contour(
    gimg, 500, init_ls, smoothing=4, balloon=-3, iter_callback=callback
)

logging.info("Done.")
plt.show()
############################################################################################################
# input()
