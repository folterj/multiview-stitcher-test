import numpy as np


class RegistrationMethod:
    def __init__(self, source_type):
        self.source_type = source_type

    def register(self, fixed_image: np.ndarray, moving_image: np.ndarray, **kwargs) -> dict:
        pass
