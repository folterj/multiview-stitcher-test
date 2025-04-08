import logging
import numpy as np
from probreg import cpd
from sklearn.neighbors import KDTree
from spatial_image import SpatialImage

from src.metrics import calc_match_metrics
from src.registration_methods.RegistrationMethod import RegistrationMethod
from src.image.util import detect_area_points, validate_transform, get_sim_physical_size
from src.util import points_to_3d


class RegistrationMethodCPD(RegistrationMethod):
    def detect_points(self, data0):
        data = data0.astype(self.source_type)

        area_points = detect_area_points(data)
        points = [point for point, area in area_points]
        d3points = points_to_3d(points)

        if len(points) >= 2:
            tree = KDTree(points, leaf_size=2)
            dist, ind = tree.query(points, k=2)
            nn_distance = np.median(dist[:, 1])
        else:
            nn_distance = 1

        return d3points, nn_distance

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        max_iter = kwargs.get('max_iter', 1000)

        fixed_points, nn_distance1 = self.detect_points(fixed_data)
        moving_points, nn_distance2 = self.detect_points(moving_data)
        threshold = np.mean([nn_distance1, nn_distance2])

        transform = None
        quality = 0
        if len(moving_points) > 1 and len(fixed_points) > 1:
            result_cpd = cpd.registration_cpd(moving_points, fixed_points, maxiter=max_iter)
            transformation = result_cpd.transformation
            S = transformation.scale * np.eye(3)
            R = transformation.rot
            T = np.eye(3) + np.hstack([np.zeros((3, 2)), transformation.t.reshape(-1, 1)])
            transform = T @ R @ S

            metrics = calc_match_metrics(fixed_points, moving_points, transform, threshold)
            quality = metrics['match_rate']

        if not validate_transform(transform, get_sim_physical_size(fixed_data, invert=True)):
            logging.error('Unable to find CPD registration')

        return {
            "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality  # float between 0 and 1 (if not available, set to 1.0)
        }
