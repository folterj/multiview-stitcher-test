import cv2 as cv
import numpy as np

from src.RegistrationMethod import RegistrationMethod


class RegistrationMethodDummy(RegistrationMethod):
    def registration(self, fixed_data, moving_data, **kwargs) -> dict:
        transform = cv.getRotationMatrix2D((fixed_data.shape[0] // 2, fixed_data.shape[1] // 2), 38, 1)
        transform = np.vstack([transform, [0, 0, 1]])
        transform[:, 2] += [300, 25, 0]

        return {
            "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
        }
