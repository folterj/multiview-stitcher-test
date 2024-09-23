import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from src.OmeSource import OmeSource
from src.image.color_conversion import hexrgb_to_rgba, int_to_rgba
from src.image.util import get_numpy_slicing, redimension_data
from src.util import reorder


class OmeZarrSource(OmeSource):
    """Zarr-compatible image source"""

    filename: str
    """original filename / URL"""
    levels: list
    """list of all image arrays for different sizes"""
    level_scales: list
    """list of all image (xy) scales"""
    shapes: list
    """list of image shapes"""
    chunk_shapes: list
    """list of image chunk shapes"""

    def __init__(self, filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()

        self.levels = []
        self.level_scales = []
        self.shapes = []
        self.chunk_shapes = []
        nchannels = 1
        try:
            location = parse_url(filename)
            if location is None:
                raise FileNotFoundError(f'Error parsing ome-zarr file {filename}')
            reader = Reader(location)
            # nodes may include images, labels etc
            # first node will be the image pixel data
            image_node = list(reader())[0]

            self.metadata = image_node.metadata
            # channel metadata from ome-zarr-py limited; get from root_attrs manually
            self.root_metadata = reader.zarr.root_attrs

            axes = self.metadata.get('axes', [])
            self.dimension_order = ''.join([axis.get('name') for axis in axes])

            for data in image_node.data:
                self.levels.append(data)

                xyzct = [1, 1, 1, 1, 1]
                for i, n in enumerate(data.shape):
                    xyzct_index = self.default_properties_order.index(self.dimension_order[i])
                    xyzct[xyzct_index] = n
                self.sizes_xyzct.append(xyzct)
                self.sizes.append((xyzct[0], xyzct[1]))
                self.pixel_types.append(data.dtype)
                self.pixel_nbits.append(data.dtype.itemsize * 8)
                self.level_scales.append(np.divide(self.sizes_xyzct[0][0], xyzct[0]))
                self.shapes.append(np.flip(reorder(data.shape, self.dimension_order, self.default_properties_order)))
                self.chunk_shapes.append(np.flip(reorder(data.chunksize, self.dimension_order, self.default_properties_order)))
                nchannels = xyzct[3]
        except Exception as e:
            raise FileNotFoundError(f'Read error: {e}')

        self.is_rgb = nchannels in (3, 4)

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        pixel_size = []
        position = []
        channels = []
        metadata = self.metadata
        axes = self.dimension_order

        units = [axis.get('unit', '') for axis in metadata.get('axes', [])]

        scale1 = [1] * len(metadata.get('axes'))
        position1 = [0] * len(metadata.get('axes'))
        # get pixelsize using largest/first scale
        transform = self.metadata.get('coordinateTransformations', [])
        if transform:
            for transform1 in transform[0]:
                if transform1['type'] == 'scale':
                    scale1 = transform1['scale']
                if transform1['type'] == 'translation':
                    position1 = transform1['translation']
            for axis in 'xyz':
                if axis in axes:
                    index = axes.index(axis)
                    pixel_size.append((scale1[index], units[index]))
                    position.append((position1[index], units[index]))
                else:
                    pixel_size.append((1, ''))
                    position.append((0, ''))
        nchannels = self.sizes_xyzct[0][3]
        # look for channel metadata
        for data in self.root_metadata.values():
            if isinstance(data, dict) and 'channels' in data:
                channels = data['channels'].copy()
                for channel in channels:
                    color = channel.pop('color', '')
                    if color != '':
                        if isinstance(color, int):
                            color = int_to_rgba(color)
                        else:
                            color = hexrgb_to_rgba(color)
                        channel['color'] = color
        if len(channels) == 0:
            if self.is_rgb:
                channels = [{'label': ''}]
            else:
                channels = [{'label': ''}] * nchannels
        self.source_pixel_size = pixel_size
        self.channels = channels
        self.source_mag = 0
        self.position = position

    def get_source_dask(self):
        return self.levels

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        redim = redimension_data(self.levels[level], self.dimension_order, self.get_dimension_order())
        slices = get_numpy_slicing(self.get_dimension_order(), **slicing)
        out = redim[slices]
        return out
