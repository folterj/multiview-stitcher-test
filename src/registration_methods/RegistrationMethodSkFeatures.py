# https://scikit-image.org/docs/stable/api/skimage.feature.html
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_brief.html

import cv2 as cv
import logging
import numpy as np
from skimage.feature import SIFT, match_descriptors
from sklearn.neighbors import KDTree
from spatial_image import SpatialImage

from src.image.ome_tiff_helper import save_tiff
from src.image.util import show_image, uint8_image, color_image, int2float_image, validate_transform, \
    get_sim_physical_size
from src.registration_methods.RegistrationMethod import RegistrationMethod


class RegistrationMethodSkFeatures(RegistrationMethod):
    def __init__(self, source_type):
        super().__init__(source_type)
        self.feature_model = SIFT()

    def detect_features(self, data0):
        data = data0.astype(self.source_type)

        data = uint8_image(data)
        self.feature_model.detect_and_extract(data)
        points = self.feature_model.keypoints
        desc = self.feature_model.descriptors

        image = color_image(data)
        for point in points:
            image = cv.drawMarker(image, point, (255, 0, 0), markerType=cv.MARKER_CROSS, markerSize=10)

        #show_image(image)
        save_tiff(f'{id(data)}.tiff', image)

        if len(points) >= 2:
            tree = KDTree(points, leaf_size=2)
            dist, ind = tree.query(points, k=2)
            nn_distance = np.median(dist[:, 1])
        else:
            nn_distance = 1

        return points, desc, nn_distance

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        fixed_points, fixed_desc, nn_distance1 = self.detect_features(fixed_data.data)
        moving_points, moving_desc, nn_distance2 = self.detect_features(moving_data.data)
        nn_distance = np.mean([nn_distance1, nn_distance2])

        matches = match_descriptors(fixed_desc, moving_desc)
        # TODO: ransac?

        transform = None
        if len(matches) >= 4:
            # TODO: get transform from matches
            fixed_points2 = np.float32([fixed_points[m[0]] for m in matches])
            moving_points2 = np.float32([moving_points[m[1]] for m in matches])
            transform = np.eye(3)

        if not validate_transform(transform, get_sim_physical_size(fixed_data)):
            logging.error('Not enough matches for feature-based registration')
            transform = np.eye(3)

        return {
            "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
        }
