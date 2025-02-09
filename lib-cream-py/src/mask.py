from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from PIL import Image
from numpy import ndarray

from util import image_to_array


class Mask(ABC):
    @abstractmethod
    def find_mask(self) -> ndarray:
        """
        This method returns the mask
        shape: (A, B, 3)
        values from [0.0, 255.0]
        """
        pass

    @abstractmethod
    def find_mask_simple(self):
        """
        This method returns the mask
        shape: (A, B)
        values from [0, 1] uint8
        1 = True
        0 = False
        """
        pass

    def display(self):
        return Image.fromarray(self.find_mask().astype('uint8'))


def find_mask(colored: ndarray, color):
    return mask


class ColorMask(Mask):
    def __init__(self, image: Union[Image, ndarray], rgb: Union[tuple[float, float, float], tuple[int, int, int]]):
        """
        :param image: ndarray = AxBx3 or AxBx1 or AxB. values should be in the range [0, 1]. [0, 255]
        """
        self.color = [v / 255.0 if v > 1 else v for v in rgb]
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            arr = image_to_array(image)
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

            if arr.max() > 1:
                arr = np.array(arr / 255.0)
        else:
            raise TypeError("Expected image to be a PIL Image or a NumPy ndarray")
        self.mask = self.find_mask_logic(arr)

    def find_mask_logic(self, image):
        """
        return AxB with values from [0, 1] 1 is found
        """
        m = np.zeros(image.shape[:2], np.uint8)
        i, j = np.where(np.all(image == self.color, axis=-1))

        if len(i) > 0:
            m[i, j] = 1

        return m

    def find_mask_simple(self):
        return self.mask

    def find_mask(self) -> ndarray:
        """
        shape: (A, B, 3)
        values from [0.0, 255.0] 0 is found
        """
        arr = np.stack([self.find_mask_simple()] * 3, axis=-1)
        arr = np.logical_not(arr) * 255.0
        return arr


def convert_matrix(matrix):
    if not np.all(matrix[:, :, 0] == matrix[:, :, 1]) or not np.all(matrix[:, :, 1] == matrix[:, :, 2]):
        raise ValueError("Values in all three dimensions must match")

    return matrix.mean(axis=2)


class RawMask2(ColorMask):
    def __init__(self, image: Union[Image, ndarray], upper_is_mask: bool,
                 threshold: Union[float, int]):
        super().__init__(image, (threshold, threshold, threshold))
        self.upper_is_mask = upper_is_mask

    def find_mask_logic(self, image):
        """
        return AxB with values from [0, 1] 1 is found
        """
        m = np.zeros(image.shape[:2], np.uint8)
        image = convert_matrix(image)
        i, j = np.where(np.all(image > self.color[0] if self.upper_is_mask else image < self.color[0], axis=-1))

        if len(i) > 0:
            m[i, j] = 1

        return m


class RawMask(ColorMask):
    def __init__(self, image: Union[Image, ndarray], mask_black=True):
        """
        :param image: ndarray = AxBx3 or AxBx1 or AxB. values should be in the range [0, 1]
        """
        super().__init__(image, (0, 0, 0) if mask_black else (1, 1, 1))


if __name__ == '__main__':
    img = Image.open('../../decensor_input/mermaid_censored.png')
    mask = ColorMask(img, (0, 1, 0)).display()
    mask2 = RawMask(mask, mask_black=True).find_mask()
