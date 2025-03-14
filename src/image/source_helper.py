import os

from src.OmeZarrSource import OmeZarrSource
from src.TiffSource import TiffSource
from src.util import get_filetitle


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
    stats = ''
    for filename in filenames:
        source = create_source(filename)
        position = source.get_position_micrometer()
        rotation = source.get_rotation()
        if rotation is None:
            rotation = ''
        stats += f'{get_filetitle(filename)} {position} {rotation}\n'
    return stats
