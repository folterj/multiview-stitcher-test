import numpy as np
import os

from src.OmeZarrSource import OmeZarrSource
from src.TiffSource import TiffSource
from src.util import get_filetitle, get_orthogonal_pairs


def create_source(filename):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        source = TiffSource(filename)
    elif ext.startswith('zar'):
        source = OmeZarrSource(filename)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    return source


def get_images_metadata(filenames):
    summary = 'Filename\tSize\tPosition\tRotation\n'
    sizes = []
    centers = []
    rotations = []
    for filename in filenames:
        source = create_source(filename)
        size = source.get_physical_size_micrometer()
        sizes.append(size)
        position = source.get_position_micrometer()
        rotation = source.get_rotation()
        rotations.append(rotation)

        summary += f'{get_filetitle(filename)}\t{tuple(size)}\t{tuple(position)}'
        if rotation is not None:
            summary += f'\t{rotation}'
        summary += '\n'

        if len(size) < len(position):
            size = list(size) + [0]
        center = np.array(position) + np.array(size) / 2
        centers.append(center)

    rotations2 = []
    for rotation, size in zip(rotations, sizes):
        if rotation is None:
            _, angles = get_orthogonal_pairs(centers, size)
            rotation = -np.mean(angles)
        rotations2.append(rotation)
    return {'center': np.mean(centers, 0),
            'rotation': np.mean(rotations2),
            'summary': summary}
