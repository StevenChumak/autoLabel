import cv2
import matplotlib.pyplot as plt
from utils import utils

raillabel_path = r"C:\Users\St\Desktop\Git\raillabel"

img = cv2.imread("001_prediction.png")  # read image
img_copy = img.copy()

# colors used to color different classes
ego_color = [137, 49, 239]  # left_trackbed     | Blue-Violet
ego_rail_color = [242, 202, 25]  # left_rails        | Jonquil oder auch gelb
neighbor_color = [225, 24, 69]  # ego_trackbed      | Spanish Crimson

# convert colorized labal to class
# TODO: dont colorize the classes and let NN output the classes only?
trackbed_data = utils.mask_to_class(img, color=ego_color)
rail_data = utils.mask_to_class(img, color=ego_rail_color)

# TODO: do I need this?
trackbed_img = cv2.cvtColor(trackbed_data, cv2.COLOR_BGR2GRAY)
rail_img = cv2.cvtColor(rail_data, cv2.COLOR_BGR2GRAY)

skeleton_rails = utils.skel_close_dila(rail_img)
skeleton_trackbed = utils.skel_close_dila(trackbed_img)

left, right = utils.rail_seperation(skeleton_rails, skeleton_trackbed)

spline1 = utils.get_spline(left["x"], left["y"])
spline2 = utils.get_spline(right["x"], right["y"])

# plt.imshow(seperated, cmap=plt.cm.gray)
plt.plot(left["x"], left["y"], "g", lw=3)
plt.plot(right["x"], right["y"], "r", lw=3)

plt.plot(spline1["x"], spline1["y"], ":b", lw=3)
plt.plot(spline2["x"], spline2["y"], "--g", lw=3)


# fig.tight_layout()
plt.show()
