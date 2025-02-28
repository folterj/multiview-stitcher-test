import cv2 as cv
import logging
import numpy as np
from sklearn.neighbors import KDTree

from src.RegistrationMethod import RegistrationMethod


class RegistrationMethodFeatures(RegistrationMethod):
    def __init__(self, source_type):
        super().__init__(source_type)
        self.feature_model = cv.ORB_create()

    def detect_features(self, data):
        kp, desc = self.feature_model.detectAndCompute(data, None)
        points = [kp1.pt for kp1 in kp]
        if len(points) >= 2:
            tree = KDTree(points, leaf_size=2)
            dist, ind = tree.query(points, k=2)
            nn_distance = np.median(dist[:, 1])
        else:
            nn_distance = 1
        # image = cv.drawKeypoints(data, kp, data)
        # show_image(image)
        return points, desc, nn_distance

    def registration(self, fixed_data, moving_data, **kwargs) -> dict:
        fixed_points, fixed_desc, nn_distance1 = self.detect_features(fixed_data.data)
        moving_points, moving_desc, nn_distance2 = self.detect_features(moving_data.data)
        nn_distance = np.mean([nn_distance1, nn_distance2])

        matcher = cv.BFMatcher()

        # matches = matcher.match(fixed_desc, moving_desc)
        matches0 = matcher.knnMatch(fixed_desc, moving_desc, k=2)
        matches = []
        for m, n in matches0:
            if m.distance < 0.75 * n.distance:
                matches.append(m)

        if len(matches) >= 4:
            fixed_points2 = np.float32([fixed_points[m.queryIdx] for m in matches])
            moving_points2 = np.float32([moving_points[m.trainIdx] for m in matches])
            transform, mask = cv.findHomography(fixed_points2, moving_points2, method=cv.USAC_MAGSAC,
                                                ransacReprojThreshold=nn_distance)
        else:
            logging.error('Not enough matches for feature-based registration')
            transform = np.eye(3)

        return {
            "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
        }
