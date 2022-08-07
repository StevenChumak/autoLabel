import glob
import math
import os

import cv2
import numpy as np
from numpy.polynomial import Laguerre

# dir = r"/home/st/Desktop/best_images/"
dir = r"C:\Users\St\Desktop\best_images"
glob_var = os.path.join(dir, "*_prediction.png")

predictions = glob.glob(glob_var)
predictions = sorted(predictions)

assert len(predictions) > 0, "No images found at:\n\t{}".format(dir)

color = [242, 202, 25]


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


import shutil

for img in predictions:
    # shutil.copy2(img, r"C:\Users\St\Desktop\test_dataset\images")
    img_data = cv2.imread(img)
    img_array = np.array(img_data)

    rail_data = mask_to_class(img_array, color=color)
    rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)

    import hdbscan
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans, MiniBatchKMeans

    blue = img_array[:, :, 0]
    red = img_array[:, :, 1]
    green = img_array[:, :, 2]
    original_shape = red.shape

    samples = np.column_stack([red.flatten(), green.flatten(), blue.flatten()])

    image_gray = rail_img.reshape(rail_img.shape[0] * rail_img.shape[1], 1)

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    # cluster_labels = clusterer.fit_predict(image_gray).reshape(rail_img.shape)

    # clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
    # cluster_labels = clusterer.fit_predict(samples).reshape(rail_img.shape)

    # clf = KMeans(n_clusters=3)
    # kmeans = clf.fit_predict(samples).reshape(rail_img.shape)

    from skimage.morphology import closing, dilation, erosion, skeletonize

    close = closing(rail_img // 255)
    dila = dilation(rail_img // 255)
    skeleton = skeletonize(rail_img // 255)

    dila_close = closing(dila)
    close_dila = dilation(close)

    skeleton_close = skeletonize(close)
    skeleton_dila = skeletonize(dila)

    skeleton_dila_close = skeletonize(dila_close)
    skeleton_close_dila = skeletonize(close_dila)

    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(rail_img)  # , cmap=plt.cm.gray)
    ax[0].set_title("Original")
    ax[1].imshow(close)  # , cmap=plt.cm.gray)
    ax[1].set_title("Pure closing")
    ax[2].imshow(dila)  # , cmap=plt.cm.gray)
    ax[2].set_title("Pure dilatation")

    ax[4].imshow(dila_close)  # , cmap=plt.cm.gray)
    ax[4].set_title("dila_close")
    ax[5].imshow(close_dila)  # , cmap=plt.cm.gray)
    ax[5].set_title("close_dila")

    # ax[6].imshow(skeleton_dila)#, cmap=plt.cm.gray)
    # ax[6].set_title("Skele-dila")
    ax[7].imshow(skeleton_close)  # , cmap=plt.cm.gray)
    ax[7].set_title("skeleton_close")
    ax[8].imshow(skeleton_dila)  # , cmap=plt.cm.gray)
    ax[8].set_title("skeleton_dila")

    # ax[9].imshow(skeleton_eros_dila)#, cmap=plt.cm.gray)
    # ax[9].set_title("skeleton_eros_dila")
    ax[10].imshow(skeleton_dila_close)  # , cmap=plt.cm.gray)
    ax[10].set_title("skeleton_dila_close")
    ax[11].imshow(skeleton_close_dila)  # , cmap=plt.cm.gray)
    ax[11].set_title("skeleton_close_dila")

    fig.tight_layout()
    plt.show()
