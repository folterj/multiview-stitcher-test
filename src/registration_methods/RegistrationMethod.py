from abc import ABC, abstractmethod
from spatial_image import SpatialImage


class RegistrationMethod(ABC):
    def __init__(self, source_type):
        self.source_type = source_type

    @abstractmethod
    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        # this returns the transform in pixel space
        # reg_func_transform = linalg.inv(params_transform) / spacing
        # params_transform = linalg.inv(reg_func_transform * spacing)
        return {
            "affine_matrix": [],  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
        }
