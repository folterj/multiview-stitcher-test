import frc
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from sklearn.metrics import euclidean_distances

from src.image.util import image_reshape
from src.util import apply_transform


def calc_match_metrics(points1, points2, transform, threshold, lowe_ratio=None):
    metrics = {}
    transformed_points1 = apply_transform(points1, transform)
    npoints1, npoints2 = len(points1), len(points2)
    npoints = min(npoints1, npoints2)
    if npoints1 == 0 or npoints2 == 0:
        return metrics

    swapped = (npoints1 > npoints2)
    if swapped:
        points1, points2 = points2, points1

    distance_matrix = euclidean_distances(transformed_points1, points2)
    matching_distances = np.diag(distance_matrix)
    if npoints1 == npoints2 and np.mean(matching_distances < threshold) > 0.5:
        # already matching points lists
        nmatches = np.sum(matching_distances < threshold)
    else:
        matches = []
        distances0 = []
        for rowi, row in enumerate(distance_matrix):
            sorted_indices = np.argsort(row)
            index0 = sorted_indices[0]
            distance0 = row[index0]
            matches.append((rowi, sorted_indices))
            distances0.append(distance0)
        sorted_matches = np.argsort(distances0)

        done = []
        nmatches = 0
        matching_distances = []
        for sorted_match in sorted_matches:
            i, match = matches[sorted_match]
            for ji, j in enumerate(match):
                if j not in done:
                    # found best, available match
                    distance0 = distance_matrix[i, j]
                    distance1 = distance_matrix[i, match[ji + 1]] if ji + 1 < len(match) else np.inf
                    matching_distances.append(distance0)    # use all distances to also weigh in the non-matches
                    if distance0 < threshold and (lowe_ratio is None or distance0 < lowe_ratio * distance1):
                        done.append(j)
                        nmatches += 1
                    break

    metrics['nmatches'] = nmatches
    metrics['match_rate'] = nmatches / npoints if npoints > 0 else 0
    distance = np.mean(matching_distances) if nmatches > 0 else np.inf
    metrics['distance'] = float(distance)
    metrics['norm_distance'] = float(distance / threshold)
    return metrics


def calc_frc(image1, image2):
    pixel_size1 = si_utils.get_spacing_from_sim(image1)
    pixel_size2 = si_utils.get_spacing_from_sim(image2)
    pixel_size = np.mean([pixel_size1['x'], pixel_size1['y'], pixel_size2['x'], pixel_size2['y']])
    image1 = image1.squeeze()
    image2 = image2.squeeze()
    max_size = np.flip(np.max([image1.shape, image2.shape], 0))
    image1 = frc.util.square_image(image_reshape(image1, max_size), add_padding=True)
    image2 = frc.util.square_image(image_reshape(image2, max_size), add_padding=True)

    frc_curve = frc.two_frc(image1, image2)
    xs_pix = np.arange(len(frc_curve)) / max(max_size)
    # scale has units [pixels <length unit>^-1] corresponding to original image
    xs_nm_freq = xs_pix / pixel_size
    frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, max_size)
    #plt.plot(xs_nm_freq, thres(xs_nm_freq))
    #plt.plot(xs_nm_freq, frc_curve)
    #plt.show()
    return frc_res
