import pathlib
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utility import image_approx, utils

shutil.rmtree("fig", ignore_errors=True)

raillabel_path = r"C:\Users\St\Desktop\Git\raillabel"

images = [
    r"C:\Users\St\Desktop\data\images\nlb_summer_000040.png",
    r"C:\Users\St\Desktop\data\images\nlb_summer_000041.png",
    r"C:\Users\St\Desktop\data\images\nlb_summer_000042.png",
    r"C:\Users\St\Desktop\data\images\nlb_summer_000043.png",
    r"C:\Users\St\Desktop\data\images\nlb_summer_000044.png",
]

masks = [
    r"C:\Users\St\Desktop\data\masks\nlb_summer_000040.png",
    r"C:\Users\St\Desktop\data\masks\nlb_summer_000041.png",
    r"C:\Users\St\Desktop\data\masks\nlb_summer_000042.png",
    r"C:\Users\St\Desktop\data\masks\nlb_summer_000043.png",
    r"C:\Users\St\Desktop\data\masks\nlb_summer_000044.png",
]


def trainID(mask):
    holder = np.zeros((mask.shape[0], mask.shape[1]))
    for id, value in zip([48, 49, 50, 51, 52, 53], [3, 4, 1, 2, 3, 4]):
        holder += np.where(mask == id, value, 0)

    return np.array(holder, dtype=np.uint8)


def colorize_mask(image_array, color_mapping):
    """
    Colorize the segmentation mask
    """
    image_array = trainID(image_array)
    new_mask = Image.fromarray(image_array.astype(np.uint8)).convert("P")
    new_mask.putpalette(color_mapping)
    return new_mask


def fill_colormap():
    palette = [
        0,
        0,
        0,  # background        | schwarrz
        137,
        49,
        239,  # left_trackbed     | Blue-Violet
        242,
        202,
        25,  # left_rails        | Jonquil oder auch gelb
        225,
        24,
        69,  # ego_trackbed      | Spanish Crimson
        0,
        87,
        233,  # ego_rails         | RYB Blue
        135,
        233,
        17,  # right_trackbed    | Alien Armpit oder grün
        255,
        0,
        189,  # right_rails        | Shocking Pink
    ]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    color_mapping = palette
    return color_mapping


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

    return rotatedImg


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


approx_dict = []

nKnots = 20
logging = False
minDistance = 4
interpolation = "catmull-rom"

for i, data in enumerate(zip(images, masks)):
    img = cv2.imread(data[0])
    mask = cv2.imread(data[1], cv2.IMREAD_GRAYSCALE)

    color_mapping = fill_colormap()
    color_image = colorize_mask(mask, color_mapping)
    # approx = image_approx.ApproxData(image=img_path, mask=mask_path)

    height, width, channels = img.shape

    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # colors used to color different classes
    # ego_color = [137, 49, 239]  # left_trackbed     | Blue-Violet
    # ego_rail_color = [242, 202, 25]  # left_rails        | Jonquil oder auch gelb
    # neighbor_color = [225, 24, 69]  # ego_trackbed      | Spanish Crimson

    ego_trackbed_color = [50]  # left_trackbed     | Blue-Violet
    ego_rail_color = [51]  # left_rails        | Jonquil oder auch gelb
    neighbor_trackbed_color = [48, 52]  # ego_trackbed      | Spanish Crimson
    neighbor_rail_color = [49, 53]  # ego_trackbed      | Spanish Crimson
    neighbor_single = np.concatenate((neighbor_trackbed_color, neighbor_rail_color))

    # convert colorized labal to class
    # TODO: dont colorize the classes and let NN output the classes only?
    trackbed_data = utils.mask_to_class(mask, color=ego_trackbed_color, gray=True)
    rail_data = utils.mask_to_class(mask, color=ego_rail_color, gray=True)
    neighor_data = utils.mask_to_class(mask, color=neighbor_single, gray=True)

    # TODO: do I need this?
    # trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
    # rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)
    # neighor_img = cv2.cvtColor(neighor_data, cv2.COLOR_BGR2GRAY)
    # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray_mask = neighor_data

    skeleton_rails = utils.pre_process(rail_data)
    skeleton_trackbed = utils.pre_process(trackbed_data)
    neighor_trackbed = utils.pre_process(neighor_data)

    # following code is an adaptations of:
    # source: https://medium.com/swlh/image-processing-with-python-connected-components-and-region-labeling-3eef1864b951
    from skimage.measure import find_contours, label, regionprops

    # ego_x, ego_y, ego_w, ego_h = cv2.boundingRect(rail_data)
    # ego_points=np.float32([[ego_x,ego_y], [ego_x, ego_y+ego_h], [ego_x+ego_w, ego_y], [ego_x+ego_w, ego_y+ego_h]])
    # p4 = [ego_x, ego_y+ego_h]
    # p1 = [ego_x, ego_y]
    # p2 = [ego_x+ego_w, ego_y]
    # p3 = [ego_x+ego_w, ego_y+ego_h]
    # w1 = int(np.linalg.norm(np.array(p2) - np.array(p3)))
    # w2 = int(np.linalg.norm(np.array(p4) - np.array(p1)))
    # h1 = int(np.linalg.norm(np.array(p1) - np.array(p2)))
    # h2 = int(np.linalg.norm(np.array(p3) - np.array(p4)))
    # maxWidth = max(w1, w2)
    # maxHeight = max(h1, h2)
    # colors=[[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]]
    # for point, color in zip(ego_points, colors):
    #     img = cv2.circle(img, point, 3, color, 3)
    # img = cv2.rectangle(img,(ego_x,ego_y),(ego_x+ego_w,ego_y+ego_h),(0,255,0),2)
    # ego_contours, _ = cv2.findContours(rail_data, 1, 1)
    # ego_rect = cv2.minAreaRect(ego_contours[0])
    # ego_box = np.array(cv2.boxPoints(ego_rect), dtype=np.int0)

    gray_mask = utils.pre_process(gray_mask, fast=False)
    label_im = label(gray_mask, connectivity=2)
    regions = regionprops(label_im, cache=True)

    mask_list = []
    bbox = []
    list_of_index = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if area > 2000:
            mask_list.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
    count = len(mask_list)
    margin = 10
    import numpy as np

    # fig, ax = plt.subplots(2, int(count//2), figsize=(15,8))
    test_img = img.copy()
    counter = 0

    test_path = pathlib.Path("./test/{}/".format(i))
    test_path.mkdir(parents=True, exist_ok=True)

    for box, region_mask in zip(bbox, mask_list):
        # for axis, box, mask in zip(ax.flatten(), bbox, masks):
        # clip box points
        y1, y_min_cutoff = (
            (box[0] - margin, 0) if 0 < box[0] - margin else (0, abs(box[0] - margin))
        )
        y2, y_max_cutoff = (
            (box[2] + margin, 0)
            if img.shape[0] > box[2] + margin
            else (img.shape[0], abs(box[2] + margin - img.shape[0]))
        )

        x1, x_min_cutoff = (
            (box[1] - margin, 0) if 0 < box[1] - margin else (0, abs(box[1] - margin))
        )
        x2, x_max_cutoff = (
            (box[3] + margin, 0)
            if img.shape[1] > box[3] + margin
            else (img.shape[1], abs(box[3] + margin - img.shape[1]))
        )

        x1 = box[1] - margin if 0 < box[1] - margin else 0
        x2 = box[3] + margin if img.shape[1] > box[3] + margin else img.shape[1]

        position = "left" if (box[1] + box[3]) < width else "right"

        cv2.imwrite(
            str(test_path / "{}_mask.png".format(counter)),
            np.array(region_mask * 255, dtype=np.uint8),
        )
        small = False
        if not small:
            old_region_mask = np.pad(region_mask, margin)
            region_mask = expand_binMask(region_mask, n=margin)
            cv2.imwrite(
                str(test_path / "{}_mask_expanded.png".format(counter)),
                np.array(region_mask * 255, dtype=np.uint8),
            )

            red = img[:, :, 0][y1:y2, x1:x2]
            green = img[:, :, 1][y1:y2, x1:x2]
            blue = img[:, :, 2][y1:y2, x1:x2]

            neighbor_slice = gray_mask[y1:y2, x1:x2]
            # if clipping was done, clip mask too
            region_mask = region_mask[
                y_min_cutoff : region_mask.shape[0] - y_max_cutoff,
                x_min_cutoff : region_mask.shape[1] - x_max_cutoff,
            ]
            old_region_mask = old_region_mask[
                y_min_cutoff : old_region_mask.shape[0] - y_max_cutoff,
                x_min_cutoff : old_region_mask.shape[1] - x_max_cutoff,
            ]
            neighbor_slice_masked = neighbor_slice * region_mask
        else:
            old_region_mask = region_mask
            red = img[:, :, 0][box[0] : box[2], box[1] : box[3]]
            green = img[:, :, 1][box[0] : box[2], box[1] : box[3]]
            blue = img[:, :, 2][box[0] : box[2], box[1] : box[3]]

            neighbor_slice = gray_mask[box[0] : box[2], box[1] : box[3]]
            neighbor_slice_masked = neighbor_slice * region_mask

        image = np.uint8(np.dstack([red, green, blue]))
        region_mask = np.uint8(region_mask)

        red_slice = image[:, :, 0] * region_mask
        green_slice = image[:, :, 1] * region_mask
        blue_slice = image[:, :, 2] * region_mask
        image_slice = np.uint8(np.dstack([red_slice, green_slice, blue_slice]))

        # find contours / rectangle
        neighbor_contours, _ = cv2.findContours(region_mask, 1, 1)
        neighbor_rect = cv2.minAreaRect(neighbor_contours[0])

        # crop
        img_croped = rotateAndScale(image, neighbor_rect, position)

        # transform
        localized_mask = np.uint8(label_im == counter + 1)
        x, y, w, h = cv2.boundingRect(localized_mask)

        p1 = [x, y + h]
        p4 = [x, y]
        p3 = [x + w, y]
        p2 = [x + w, y + h]

        w1 = int(np.linalg.norm(np.array(p2) - np.array(p3)))
        w2 = int(np.linalg.norm(np.array(p4) - np.array(p1)))
        h1 = int(np.linalg.norm(np.array(p1) - np.array(p2)))
        h2 = int(np.linalg.norm(np.array(p3) - np.array(p4)))

        maxWidth = max(w1, w2)
        maxHeight = max(h1, h2)
        neighbor_points = [p1, p2, p3, p4]

        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # colors=[[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]]
        # for point, color in zip(np.uint32(neighbor_points), colors):
        #     img = cv2.circle(img, point, 3, color, 3)

        output_poins = np.float32(
            [
                [0, 0],
                [0, maxHeight - 1],
                [maxWidth - 1, maxHeight - 1],
                [maxWidth - 1, 0],
            ]
        )

        matrix = cv2.getPerspectiveTransform(np.float32(neighbor_points), output_poins)
        result = cv2.warpPerspective(
            mask, matrix, (maxWidth, maxHeight), cv2.INTER_LINEAR
        )
        # if position == "right":
        #     result = cv2.flip(result, 1)

        # matrix = cv2.getPerspectiveTransform(neighbor_points, ego_points)
        # result = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))

        region_mask_red = region_mask * 255
        region_mask_green = np.zeros([region_mask.shape[0], region_mask.shape[1]])
        region_mask_blue = old_region_mask * 255
        region_mask_stacked = np.array(
            np.dstack([region_mask_red, region_mask_green, region_mask_blue]),
            dtype=np.uint8,
        )

        added = cv2.addWeighted(image, 1, region_mask_stacked, 0.5, 0)
        cv2.imwrite(str(test_path / "{}_neighbor_mask.png".format(counter)), added)
        cv2.imwrite(
            str(test_path / "{}_masks_stacked.png".format(counter)), region_mask_stacked
        )
        cv2.imwrite(str(test_path / "{}_neighbor.png".format(counter)), image)
        cv2.imwrite(
            str(test_path / "{}_neighbor_slice.png".format(counter)), image_slice
        )
        cv2.imwrite(
            str(test_path / "{}_neighbor_slice_rotated.png".format(counter)), img_croped
        )
        cv2.imwrite(
            str(test_path / "{}_neighbor_slice_transformed.png".format(counter)), result
        )
        counter += 1

        cv2.rectangle(test_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # axis.imshow(image)

    # fig.show()
    rgb_mask = np.zeros_like(label_im)
    for x in list_of_index:
        rgb_mask += (label_im == x + 1).astype(int)
    red = img[:, :, 0] * rgb_mask
    green = img[:, :, 1] * rgb_mask
    blue = img[:, :, 2] * rgb_mask
    image = np.dstack([red, green, blue])
    cv2.imwrite(str(test_path / "extracted_neighbors.png"), image)
    cv2.imwrite(str(test_path / "neighbors_bb.png"), test_img)
    color_image.save(str(test_path / "original_mask.png"))
