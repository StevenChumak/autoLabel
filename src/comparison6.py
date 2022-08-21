import pathlib
import shutil
from cProfile import label

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utility import utils

shutil.rmtree("fig", ignore_errors=True)

raillabel_path = r"C:\Users\St\Desktop\Git\raillabel"

img = cv2.imread("001_input.png")  # read image
mask_img = cv2.imread("001_prediction.png")  # read mask
# mask_img = cv2.imread("bad_pred.png")  # read image

height, width, channels = img.shape

if mask_img.shape != img.shape:
    mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]))

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
trackbed_data = utils.mask_to_class(mask_img, color=ego_color)
rail_data = utils.mask_to_class(mask_img, color=ego_rail_color)
neighor_data = utils.mask_to_class(mask_img, color=neighbor_color)

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)
neighor_img = cv2.cvtColor(neighor_data, cv2.COLOR_BGR2GRAY)
gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

skeleton_rails = utils.pre_process(rail_img, open=False)
skeleton_trackbed = utils.pre_process(trackbed_img, open=True)
neighor_trackbed = utils.pre_process(neighor_img, open=True)

# following code is an adaptations of:
# source: https://medium.com/swlh/image-processing-with-python-connected-components-and-region-labeling-3eef1864b951
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import convex_hull_image

neighor_img = utils.pre_process(gray_mask, open=False, fast=False)
label_im = label(neighor_img, connectivity=2)
regions = regionprops(label_im, cache=True)

masks = []
bbox = []
list_of_index = []
for num, x in enumerate(regions):
    area = x.area
    convex_area = x.convex_area
    if num != 0 and (area > 2000):
        masks.append(regions[num].convex_image)
        bbox.append(regions[num].bbox)
        list_of_index.append(num)
count = len(masks)
margin = 10
import numpy as np

fig, ax = plt.subplots(2, int(count // 2), figsize=(15, 8))
test_img = img.copy()
counter = 0

test_path = pathlib.Path("./test/").mkdir(parents=True, exist_ok=True)


def rotateAndScale(img, rect, position, scaleFactor=0.5):
    # edited version of source: https://stackoverflow.com/a/33247373 by Luke
    # last accessed on 17.08.22

    angle = -rect[2] if position == "left" else 90 - rect[2]
    (oldY, oldX) = img.shape[
        0:2
    ]  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(
        center=(oldX / 2, oldY / 2), angle=angle, scale=scaleFactor
    )  # rotate about center of image.

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(angle)
    newX, newY = (
        abs(np.sin(r) * newY) + abs(np.cos(r) * newX),
        abs(np.sin(r) * newX) + abs(np.cos(r) * newY),
    )

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[
        0, 2
    ] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
    # if position == "left":
    rotatedImg = cv2.flip(rotatedImg, 1)

    return rotatedImg


for axis, box, mask in zip(ax.flatten(), bbox, masks):

    def expand_binMask(mask, n=5):
        padded = np.pad(mask, n)
        padded2 = np.pad(mask, n)
        for i in range(0, padded.shape[0]):
            dim1_min = i - n if i - n > 0 else 0
            dim1_max = i + n if i + n < padded.shape[0] else padded.shape[0]
            for j in range(0, padded.shape[1]):
                if padded[i][j] == True:
                    dim2_min = j - n if 0 < j - n else 0
                    dim2_max = j + n if j + n < padded.shape[1] else padded.shape[1]
                    padded2[dim1_min:dim1_max, dim2_min:dim2_max] = True
        return padded2

    y1 = box[0] - margin if 0 < box[0] - margin else 0
    y2 = box[2] + margin if img.shape[0] > box[2] + margin else img.shape[0]

    x1 = box[1] - margin if 0 < box[1] - margin else 0
    x2 = box[3] + margin if img.shape[1] > box[3] + margin else img.shape[1]

    position = "left" if (box[1] + box[3]) > width else "right"

    cv2.imwrite("test/mask{}.png".format(counter), np.array(mask * 255, dtype=np.uint8))
    small = False
    if not small:
        old_mask = np.pad(mask, margin)
        mask = expand_binMask(mask, n=margin)
        cv2.imwrite("test/mask2.png", np.array(mask * 255, dtype=np.uint8))

        red = img[:, :, 0][y1:y2, x1:x2]
        green = img[:, :, 1][y1:y2, x1:x2]
        blue = img[:, :, 2][y1:y2, x1:x2]

        neighbor_slice = neighor_img[y1:y2, x1:x2]
        neighbor_slice_masked = neighbor_slice * mask
    else:
        red = img[:, :, 0][box[0] : box[2], box[1] : box[3]]
        green = img[:, :, 1][box[0] : box[2], box[1] : box[3]]
        blue = img[:, :, 2][box[0] : box[2], box[1] : box[3]]

        neighbor_slice = neighor_img[box[0] : box[2], box[1] : box[3]]
        neighbor_slice_masked = neighbor_slice * mask

    image = np.array(np.dstack([red, green, blue]), dtype=np.uint8)
    red = image[:, :, 0] * mask
    green = image[:, :, 1] * mask
    blue = image[:, :, 2] * mask
    image_slice = np.dstack([red, green, blue])

    mask = np.array(mask, dtype=np.uint8)

    # find contours / rectangle
    contours, _ = cv2.findContours(mask, 1, 1)
    rect = cv2.minAreaRect(contours[0])

    # crop
    img_croped = rotateAndScale(image_slice, rect, position)

    mask_red = mask * 255
    mask_green = np.zeros(
        [neighbor_slice_masked.shape[0], neighbor_slice_masked.shape[1]]
    )
    mask_blue = old_mask * 255
    mask_stacked = np.array(
        np.dstack([mask_red, mask_green, mask_blue]), dtype=np.uint8
    )

    added = cv2.addWeighted(image, 1, mask_stacked, 0.5, 0)
    cv2.imwrite("test/neighbor_slice.png", neighbor_slice)
    cv2.imwrite("test/added.png", added)
    cv2.imwrite("test/mask_stacked.png", mask_stacked)

    cv2.imwrite("test/img{}.png".format(counter), image)
    cv2.imwrite("test/img_slice{}.png".format(counter), image_slice)
    cv2.imwrite("test/img_slice_rotated{}.png".format(counter), img_croped)
    counter += 1

    # cv2.rectangle(test_img, (x1,y1),(x2,y2),(255,255,0),2)

    axis.imshow(image)

fig.show()
rgb_mask = np.zeros_like(label_im)
for x in list_of_index:
    rgb_mask += (label_im == x + 1).astype(int)
red = img[:, :, 0] * rgb_mask
green = img[:, :, 1] * rgb_mask
blue = img[:, :, 2] * rgb_mask
image = np.dstack([red, green, blue])
cv2.imwrite("test/test.png", image)
cv2.imwrite("test/test_bb.png", test_img)

input()

# contours = find_contours(gray_mask)

# fig, ax = plt.subplots()
# ax.imshow(gray_mask, cmap=plt.cm.gray)
# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(10, 10))
# ax = axes.ravel()

# ax[0].imshow(gray_mask, cmap=plt.cm.gray)
# ax[0].set_title("gray_mask")

# ax[1].imshow(gray_mask_processed, cmap=plt.cm.gray)
# ax[1].set_title("gray_mask_processed")

# ax[2].imshow(label_im)
# ax[2].set_title("labeled")

# fig.tight_layout()
# fig.savefig("fig.png")
# # fig.show()
# input()
# x, y, w, h = cv2.boundingRect(neighor_img)

# left, right, left_IImage, right_IImage = utils.rail_seperation(
#     skeleton_rails, skeleton_trackbed
# )
# thicc_left, thicc_right, thicc_left_IImage, thicc_right_IImage = utils.rail_seperation(
#     rail_img, skeleton_trackbed
# )


# from skimage.measure import approximate_polygon

# test = approximate_polygon((left["xy"]), tolerance=4)

# approx_left, lin_left = utils.approximateKnots(
#     left_IImage,
#     nKnots=nKnots,
#     minDistance=minDistance,
#     interpolation=interpolation,
#     log=logging,
#     direction="left",
# )
# approx_right, lin_right = utils.approximateKnots(
#     right_IImage,
#     nKnots=nKnots,
#     minDistance=minDistance,
#     interpolation=interpolation,
#     log=logging,
#     direction="right",
# )
