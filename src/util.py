import ast
import cv2 as cv
import glob
import math
import numpy as np
import os
import re
from scipy.spatial.transform import Rotation


def get_default(x, default):
    return default if x is None else x


def ensure_list(x) -> list:
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    else:
        return [x]


def reorder(items: list, old_order: str, new_order: str, default_value: int = 0) -> list:
    new_items = []
    for label in new_order:
        if label in old_order:
            item = items[old_order.index(label)]
        else:
            item = default_value
        new_items.append(item)
    return new_items


def filter_dict(dict0: dict) -> dict:
    new_dict = {}
    for key, value0 in dict0.items():
        if value0 is not None:
            values = []
            for value in ensure_list(value0):
                if isinstance(value, dict):
                    value = filter_dict(value)
                values.append(value)
            if len(values) == 1:
                values = values[0]
            new_dict[key] = values
    return new_dict


def desc_to_dict(desc: str) -> dict:
    desc_dict = {}
    if desc.startswith('{'):
        try:
            metadata = ast.literal_eval(desc)
            return metadata
        except:
            pass
    for item in re.split(r'[\r\n\t|]', desc):
        item_sep = '='
        if ':' in item:
            item_sep = ':'
        if item_sep in item:
            items = item.split(item_sep)
            key = items[0].strip()
            value = items[1].strip()
            for dtype in (int, float, bool):
                try:
                    value = dtype(value)
                    break
                except:
                    pass
            desc_dict[key] = value
    return desc_dict


def print_dict(dct: dict, indent: int = 0) -> str:
    s = ''
    if isinstance(dct, dict):
        for key, value in dct.items():
            s += '\n'
            if not isinstance(value, list):
                s += '\t' * indent + str(key) + ': '
            if isinstance(value, dict):
                s += print_dict(value, indent=indent + 1)
            elif isinstance(value, list):
                for v in value:
                    s += print_dict(v)
            else:
                s += str(value)
    else:
        s += str(dct)
    return s


def print_hbytes(nbytes: int) -> str:
    exps = ['', 'K', 'M', 'G', 'T']
    div = 1024
    exp = 0

    while nbytes > div:
        nbytes /= div
        exp += 1
    return f'{nbytes:.1f}{exps[exp]}B'


def check_round_significants(a: float, significant_digits: int) -> float:
    rounded = round_significants(a, significant_digits)
    if a != 0:
        dif = 1 - rounded / a
    else:
        dif = rounded - a
    if abs(dif) < 10 ** -significant_digits:
        return rounded
    return a


def round_significants(a: float, significant_digits: int) -> float:
    if a != 0:
        round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
        return round(a, round_decimals)
    return a


def get_filetitle(filename: str) -> str:
    filebase = os.path.basename(filename)
    title = os.path.splitext(filebase)[0].rstrip('.ome')
    return title


def dir_regex(pattern):
    files = []
    for pattern_item in ensure_list(pattern):
        files.extend(glob.glob(pattern_item, recursive=True))
    files_sorted = sorted(files, key=lambda file: find_all_numbers(get_filetitle(file)))
    return files_sorted


def find_all_numbers(text: str) -> list:
    return list(map(int, re.findall(r'\d+', text)))


def split_underscore_numeric(text: str) -> dict:
    num_parts = {}
    parts = text.split('_')
    for part in parts:
        num_span = re.search(r'\d+', part)
        if num_span:
            index = num_span.start()
            if index > 0:
                label = part[:index]
                num_parts[label] = num_span.group()
    return num_parts


def split_num_text(text: str) -> list:
    num_texts = []
    block = ''
    is_num0 = None
    if text is None:
        return None

    for c in text:
        is_num = (c.isnumeric() or c == '.')
        if is_num0 is not None and is_num != is_num0:
            num_texts.append(block)
            block = ''
        block += c
        is_num0 = is_num
    if block != '':
        num_texts.append(block)

    num_texts2 = []
    for block in num_texts:
        block = block.strip()
        try:
            block = float(block)
        except:
            pass
        if block not in [' ', ',', '|']:
            num_texts2.append(block)
    return num_texts2


def split_value_unit_list(text: str) -> list:
    value_units = []
    if text is None:
        return None

    items = split_num_text(text)
    if isinstance(items[-1], str):
        def_unit = items[-1]
    else:
        def_unit = ''

    i = 0
    while i < len(items):
        value = items[i]
        if i + 1 < len(items):
            unit = items[i + 1]
        else:
            unit = ''
        if not isinstance(value, str):
            if isinstance(unit, str):
                i += 1
            else:
                unit = def_unit
            value_units.append((value, unit))
        i += 1
    return value_units


def get_value_units_micrometer(value_units0: list) -> list:
    conversions = {'nm': 1e-3, 'µm': 1, 'um': 1, 'micrometer': 1, 'mm': 1e3, 'cm': 1e4, 'm': 1e6}
    if value_units0 is None:
        return None

    values_um = []
    for value_unit in value_units0:
        if not (isinstance(value_unit, int) or isinstance(value_unit, float)):
            value_um = value_unit[0] * conversions.get(value_unit[1], 1)
        else:
            value_um = value_unit
        values_um.append(value_um)
    return values_um


def convert_rational_value(value) -> float:
    if value is not None and isinstance(value, tuple):
        value = value[0] / value[1]
    return value


def get_moments(data, offset=(0, 0)):
    moments = cv.moments((np.array(data) + offset).astype(np.float32))    # doesn't work for float64!
    return moments


def get_moments_center(moments, offset=(0, 0)):
    return np.array([moments['m10'], moments['m01']]) / moments['m00'] + np.array(offset)


def get_center(data, offset=(0, 0)):
    moments = get_moments(data, offset=offset)
    if moments['m00'] != 0:
        center = get_moments_center(moments)
    else:
        center = np.mean(data, 0).flatten()  # close approximation
    return center.astype(np.float32)


def create_transform0(center=(0, 0), angle=0, scale=1, translate=(0, 0)):
    transform = cv.getRotationMatrix2D(center[:2], angle, scale)
    transform[:, 2] += translate
    if len(transform) == 2:
        transform = np.vstack([transform, [0, 0, 1]])   # create 3x3 matrix
    return transform


def create_transform(center, angle):
    if len(center) == 2:
        center = np.array(list(center) + [1])
    r = Rotation.from_euler('z', angle, degrees=True)
    t = center - r.apply(center, inverse=True)
    transform = np.transpose(r.as_matrix())
    transform[:, -1] += t
    return transform


def apply_transform(points, transform):
    new_points = []
    for point in points:
        point_len = len(point)
        if point_len == 2:
            point = list(point) + [1]
        new_point = np.dot(point, np.transpose(transform))
        if point_len == 2:
            new_point = new_point[:2]
        new_points.append(new_point)
    return new_points


def get_rotation_from_transform(transform):
    rotation = np.rad2deg(np.arctan2(transform[0][1], transform[0][0]))
    return rotation


def points_to_3d(points):
    return [list(point) + [0] for point in points]


def convert_xyz_to_dict(xyz, axes='xyz'):
    dct = {dim: value for dim, value in zip(axes, xyz)}
    return dct


def normalise_rotated_positions(positions0, rotations0, size):
    # in [xy(z)]
    positions = []
    rotations = []
    positions_centre = np.mean(positions0, 0)
    center_index = np.argmin([math.dist(position, positions_centre) for position in positions0])
    center_position = positions0[center_index]
    pairs, angles = get_orthogonal_pairs_from_tiles(positions0, size)
    if len(pairs) > 0:
        mean_angle = np.mean(angles)
        for position0, rotation in zip(positions0, rotations0):
            if rotation is None:
                rotation = -mean_angle
            transform = create_transform(center=center_position, angle=-rotation)
            position = apply_transform([position0], transform)[0]
            positions.append(position)
            rotations.append(rotation)
    else:
        positions = positions0
        rotations = rotations0
    return positions, rotations


def get_orthogonal_pairs_from_tiles(origins, image_size_um):
    """
    Get pairs of orthogonal neighbors from a list of tiles.
    Tiles don't have to be placed on a regular grid.
    """
    pairs = []
    angles = []
    for i, j in np.transpose(np.triu_indices(len(origins), 1)):
        origini = origins[i]
        originj = origins[j]
        distance = math.dist(origini, originj)
        if distance < max(image_size_um):
            pairs.append((i, j))
            vector = origini - originj
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            if distance < min(image_size_um):
                angle += 90
            while angle < -90:
                angle += 180
            while angle > 90:
                angle -= 180
            angles.append(angle)
    return pairs, angles


def retuple(chunks, shape):
    # from ome-zarr-py
    """
    Expand chunks to match shape.

    E.g. if chunks is (64, 64) and shape is (3, 4, 5, 1028, 1028)
    return (3, 4, 5, 64, 64)

    If chunks is an integer, it is applied to all dimensions, to match
    the behaviour of zarr-python.
    """

    if isinstance(chunks, int):
        return tuple([chunks] * len(shape))

    dims_to_add = len(shape) - len(chunks)
    return *shape[:dims_to_add], *chunks
