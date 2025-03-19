import cv2 as cv
import logging
import numpy as np
from multiview_stitcher import param_utils
from sklearn.neighbors import KDTree
from spatial_image import SpatialImage

from src.image.util import uint8_image, get_sim_physical_size, validate_transform
from src.metrics import calc_match_metrics
from src.registration_methods.RegistrationMethod import RegistrationMethod


class RegistrationMethodCvFeatures(RegistrationMethod):
    def __init__(self, source_type):
        super().__init__(source_type)
        self.feature_model = cv.SIFT_create(contrastThreshold=0.1)
        #self.feature_model = cv.ORB_create(patchSize=8, edgeThreshold=8)

    def detect_features(self, data0):
        data = data0.astype(self.source_type)

        data = uint8_image(data)
        scale = min(1000 / np.linalg.norm(data.shape), 1)
        data = cv.resize(data, (0, 0), fx=scale, fy=scale)
        keypoints, desc = self.feature_model.detectAndCompute(data, None)
        points = [np.array(keypoint.pt) / scale for keypoint in keypoints]

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
        threshold = np.mean([nn_distance1, nn_distance2])

        matcher = cv.BFMatcher()
        #matches0 = matcher.match(fixed_desc, moving_desc)
        matches0 = matcher.knnMatch(fixed_desc, moving_desc, k=2)

        matches = []
        for m, n in matches0:
            if m.distance < 0.92 * n.distance:
                matches.append(m)

        transform = None
        quality = 0
        if len(matches) >= 4:
            fixed_points2 = np.float32([fixed_points[match.queryIdx] for match in matches])
            moving_points2 = np.float32([moving_points[match.trainIdx] for match in matches])
            transform, inliers = cv.findHomography(fixed_points2, moving_points2,
                                                   method=cv.USAC_MAGSAC, ransacReprojThreshold=threshold)
            if transform is not None:
                fixed_points3 = [point for point, is_inlier in zip(fixed_points2, inliers) if is_inlier]
                moving_points3 = [point for point, is_inlier in zip(moving_points2, inliers) if is_inlier]
                metrics = calc_match_metrics(fixed_points3, moving_points3, transform, threshold)
                #quality = np.mean(inliers)
                quality = metrics['match_rate']

        if not validate_transform(transform, get_sim_physical_size(fixed_data, invert=True)):
            logging.error('Unable to find feature-based registration')
            transform = np.eye(3)

        return {
            "affine_matrix": param_utils.invert_coordinate_order(transform),  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality  # float between 0 and 1 (if not available, set to 1.0)
        }
