import pathlib

import cv2
import numpy as np


class ApproxData:
    def __init__(self, image=None, mask=None):
        self.image_path, self.image_name, self.image_data = self.__check_input(image)
        self.mask_path, self.mask_name, self.mask_data = self.__check_input(mask)

    def __check_input(self, data):
        if isinstance(data, str):
            image_path = pathlib.Path(data)
            if image_path.exists():
                image_name = image_path.stem
                image_data = cv2.imread(data)
            else:
                raise FileNotFoundError("Could not find {}".format(image_path))

        elif isinstance(data, np.ndarray):
            image_data = data
            image_path = None
            image_name = None

        elif data is None:
            image_data = None
            image_path = None
            image_name = None

        else:
            raise ValueError(
                "Input data should be an image array or a path to an image"
            )

        return image_path, image_name, image_data

    def add_mask(self, mask):
        self.mask_path, self.mask_name, self.mask_data = self.__check_input(mask)

    def add_image(self, image):
        self.image_path, self.image_name, self.image_data = self.__check_input(image)
