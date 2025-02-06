import dask
import dask.array as da
import logging
import numpy as np

from src.image.color_conversion import *
from src.image.util import *
from src.util import *


class OmeSource:
    """OME-compatible image source (base class)"""
    """Internal image format is [TCZYX]"""

    metadata: dict
    """metadata dictionary"""
    has_ome_metadata: bool
    """has ome metadata"""
    dimension_order: str
    """source dimension order"""
    output_dimension_order: str
    """data dimension order"""
    source_pixel_size: list
    """original source pixel size"""
    target_pixel_size: list
    """target pixel size"""
    sizes: list
    """x/y size pairs for all pages"""
    sizes_xyzct: list
    """xyzct size for all pages"""
    pixel_types: list
    """pixel types for all pages"""
    pixel_nbits: list
    """#bits for all pages"""
    channels: list
    """channel information for all image channels"""
    position: list
    """source position information"""
    rotation: float
    """source rotation information"""

    default_properties_order = 'xyzct'
    default_physical_unit = 'µm'

    def __init__(self):
        self.metadata = {}
        self.has_ome_metadata = False
        self.dimension_order = ''
        self.output_dimension_order = ''
        self.source_pixel_size = []
        self.target_pixel_size = []
        self.sizes = []
        self.sizes_xyzct = []
        self.pixel_types = []
        self.pixel_nbits = []
        self.channels = []

    def _init_metadata(self,
                       source_reference: str,
                       source_pixel_size: list = None,
                       target_pixel_size: list = None,
                       source_info_required: bool = False):

        self.source_reference = source_reference
        self.target_pixel_size = target_pixel_size
        self._find_metadata()
        if (len(self.source_pixel_size) == 0 or self.source_pixel_size[0][0] == 0
                or self.source_pixel_size[0][1] == '' or self.source_pixel_size[0][1] == 'inch') \
                and source_pixel_size is not None:
            # if pixel size is not set, or default/unspecified value
            self.source_pixel_size = source_pixel_size
        if len(self.source_pixel_size) == 0 or self.source_pixel_size[0][0] == 0:
            msg = f'{source_reference}: No source pixel size in metadata or provided'
            if source_info_required:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        self._init_sizes()

    def _get_ome_metadata(self):
        images = ensure_list(self.metadata.get('Image', {}))[0]
        pixels = images.get('Pixels', {})
        self.source_pixel_size = []
        size = float(pixels.get('PhysicalSizeX', 0))
        if size > 0:
            self.source_pixel_size.append((size, pixels.get('PhysicalSizeXUnit', self.default_physical_unit)))
        size = float(pixels.get('PhysicalSizeY', 0))
        if size > 0:
            self.source_pixel_size.append((size, pixels.get('PhysicalSizeYUnit', self.default_physical_unit)))
        size = float(pixels.get('PhysicalSizeZ', 0))
        if size > 0:
            self.source_pixel_size.append((size, pixels.get('PhysicalSizeZUnit', self.default_physical_unit)))

        position = []
        for plane in ensure_list(pixels.get('Plane', [])):
            position = []
            if 'PositionX' in plane:
                position.append((float(plane.get('PositionX')), plane.get('PositionXUnit', self.default_physical_unit)))
            if 'PositionY' in plane:
                position.append((float(plane.get('PositionY')), plane.get('PositionYUnit', self.default_physical_unit)))
            if 'PositionZ' in plane:
                position.append((float(plane.get('PositionZ')), plane.get('PositionZUnit', self.default_physical_unit)))
            #c, z, t = plane.get('TheC'), plane.get('TheZ'), plane.get('TheT')
        self.position = position

        rotation = None
        annotations = self.metadata.get('StructuredAnnotations')
        if annotations is not None:
            if not isinstance(annotations, (list, tuple)):
                annotations = [annotations]
            for annotation_item in annotations:
                for annotations2 in annotation_item.values():
                    if not isinstance(annotations2, (list, tuple)):
                        annotations2 = [annotations2]
                    for annotation in annotations2:
                        value = annotation.get('Value')
                        unit = None
                        if isinstance(value, dict) and 'Modulo' in value:
                            modulo = value.get('Modulo', {}).get('ModuloAlongZ', {})
                            unit = modulo.get('Unit')
                            value = modulo.get('Label')
                        elif isinstance(value, str) and value.lower().startswith('angle'):
                            if ':' in value:
                                value = value.split(':')[1].split()
                            elif '=' in value:
                                value = value.split('=')[1].split()
                            else:
                                value = value.split()[1:]
                            if len(value) >= 2:
                                unit = value[1]
                            value = value[0]
                        else:
                            value = None
                        if value is not None:
                            rotation = float(value)
                            if 'rad' in unit.lower():
                                rotation = np.rad2deg(rotation)
        self.rotation = rotation

        self.source_mag = 0
        objective_id = images.get('ObjectiveSettings', {}).get('ID', '')
        for objective in ensure_list(self.metadata.get('Instrument', {}).get('Objective', [])):
            if objective.get('ID', '') == objective_id:
                self.source_mag = float(objective.get('NominalMagnification', 0))
        nchannels = self.get_nchannels()
        channels = []
        for channel0 in ensure_list(pixels.get('Channel', [])):
            channel = {'label': channel0.get('Name', '')}
            color = channel0.get('Color')
            if color:
                channel['color'] = int_to_rgba(int(color))
            channels.append(channel)
        if len(channels) == 0:
            if nchannels == 3:
                channels = [{'label': ''}]
            else:
                channels = [{'label': str(channeli)} for channeli in range(nchannels)]
        self.channels = channels

    def _init_sizes(self):
        # normalise source pixel sizes
        standard_units = {'nano': 'nm', 'micro': 'µm', 'milli': 'mm', 'centi': 'cm'}
        pixel_size = []
        for pixel_size0 in self.source_pixel_size:
            pixel_size1 = check_round_significants(pixel_size0[0], 6)
            unit1 = pixel_size0[1]
            for standard_unit in standard_units:
                if unit1.lower().startswith(standard_unit):
                    unit1 = standard_units[standard_unit]
            pixel_size.append((pixel_size1, unit1))
        if 0 < len(pixel_size) < 2:
            pixel_size.append(pixel_size[0])
        self.source_pixel_size = pixel_size

        if self.target_pixel_size is None:
            self.target_pixel_size = self.source_pixel_size

        if 0 < len(self.target_pixel_size) < 2:
            self.target_pixel_size.append(self.target_pixel_size[0])

        # set source mags
        self.source_mags = [self.source_mag]
        for i, size in enumerate(self.sizes):
            if i > 0:
                mag = self.source_mag * np.mean(np.divide(size, self.sizes[0]))
                self.source_mags.append(check_round_significants(mag, 3))

        if self.target_pixel_size:
            self.best_level, self.best_factor, self.full_factor = get_best_level_factor(self, self.target_pixel_size)
        else:
            self.best_level = 0
            self.best_factor = 1

        if self.dimension_order == '':
            x, y, z, c, t = self.get_size_xyzct()
            if t > 1:
                self.dimension_order += 't'
            if c > 1:
                self.dimension_order += 'c'
            if z > 1:
                self.dimension_order += 'z'
            self.dimension_order += 'yx'

        self.output_dimension_order = 'tczyx'

    def get_source_dask(self):
        raise NotImplementedError('Implement method in subclass')

    def get_mag(self) -> float:
        mag = self.source_mag
        # get effective mag at target pixel size
        if self.target_pixel_size:
            mag *= np.mean(self.full_factor)
        return check_round_significants(mag, 3)

    def get_dimension_order(self) -> str:
        return self.output_dimension_order

    def get_physical_size(self) -> tuple:
        physical_size = []
        for size, pixel_size in zip(self.get_size_xyzct(), self.get_pixel_size()):
            physical_size.append((np.multiply(size, pixel_size[0]), pixel_size[1]))
        return tuple(physical_size)

    def get_pixel_type(self, level: int = 0) -> np.dtype:
        return self.pixel_types[level]

    def get_pixel_nbytes(self, level: int = 0) -> int:
        return self.pixel_nbits[level] // 8

    def get_channels(self) -> list:
        return self.channels

    def get_size(self) -> tuple:
        # size at target pixel size
        return tuple(np.round(np.multiply(self.sizes[self.best_level], self.best_factor)).astype(int))

    def get_size_xyzct(self) -> tuple:
        xyzct = list(self.sizes_xyzct[self.best_level]).copy()
        size = self.get_size()
        xyzct[0:2] = size
        return tuple(xyzct)

    def get_nchannels(self):
        return self.sizes_xyzct[0][3]

    def get_pixel_size(self) -> list:
        return self.target_pixel_size

    def get_pixel_size_micrometer(self):
        return get_value_units_micrometer(self.get_pixel_size())

    def get_position(self) -> list:
        return self.position

    def get_position_micrometer(self):
        return get_value_units_micrometer(self.get_position())

    def get_rotation(self) -> float:
        return self.rotation

    def get_shape(self, dimension_order: str = None, xyzct: tuple = None) -> tuple:
        shape = []
        if dimension_order is None:
            dimension_order = self.get_dimension_order()
        if xyzct is None:
            xyzct = self.get_size_xyzct()
        for dimension in dimension_order:
            index = 'xyzct'.index(dimension)
            shape.append(xyzct[index])
        return tuple(shape)

    def get_thumbnail(self, target_size: tuple, precise: bool = False) -> np.ndarray:
        size, index = get_best_size(self.sizes, target_size)
        scale = np.divide(target_size, self.sizes[index])
        new_dimension_order = 'yxc'
        image = redimension_data(self._asarray_level(index), self.get_dimension_order(), new_dimension_order, t=0, z=0)
        if precise:
            thumbnail = precise_resize(image, scale)
        else:
            thumbnail = image_resize(image, target_size)
        thumbnail_rgb = self.render(thumbnail, new_dimension_order)
        return thumbnail_rgb

    def get_channel_window(self, channeli):
        min_quantile = 0.001
        max_quantile = 0.999

        if channeli < len(self.channels) and 'window' in self.channels[channeli]:
            return self.channels[channeli].get('window')

        dtype = self.get_pixel_type()
        if dtype.kind == 'f':
            #info = np.finfo(dtype)
            start, end = 0, 1
        else:
            info = np.iinfo(dtype)
            start, end = info.min, info.max

        nsizes = len(self.sizes)
        if nsizes > 1:
            image = self._asarray_level(nsizes - 1)
            image = np.asarray(image[:, channeli:channeli+1, ...])
            min, max = get_image_quantile(image, min_quantile), get_image_quantile(image, max_quantile)
        else:
            # do not query full size image
            min, max = start, end
        window = {'start': start, 'end': end, 'min': min, 'max': max}
        if channeli < len(self.channels):
            self.channels[channeli]['window'] = window
        return window

    def render(self, image: np.ndarray, source_dimension_order: str = None, t: int = 0, z: int = 0, channels: list = []) -> np.ndarray:
        if source_dimension_order is None:
            source_dimension_order = self.get_dimension_order()
        image = redimension_data(image, source_dimension_order, 'yxc', t=t, z=z)
        total_image = None
        n = len(self.channels)
        is_rgb = (self.get_nchannels() in (3, 4) and (n <= 1 or n == 3))
        needs_normalisation = (image.dtype.itemsize == 2)

        if not is_rgb:
            tot_alpha = 0
            for channeli, channel in enumerate(self.channels):
                if not channels or channeli in channels:
                    if n == 1:
                        channel_values = image
                    else:
                        channel_values = image[..., channeli]
                    if needs_normalisation:
                        window = self.get_channel_window(channeli)
                        channel_values = normalise_values(channel_values, window['min'], window['max'])
                    else:
                        channel_values = int2float_image(channel_values)
                    new_channel_image = np.atleast_3d(channel_values)
                    color = channel.get('color')
                    if color:
                        rgba = color
                    else:
                        rgba = [1, 1, 1, 1]
                    color = rgba[:3]
                    alpha = rgba[3]
                    if alpha == 0:
                        alpha = 1
                    new_channel_image = new_channel_image * np.multiply(color, alpha).astype(np.float32)
                    if total_image is None:
                        total_image = new_channel_image
                    else:
                        total_image += new_channel_image
                    tot_alpha += alpha
            if tot_alpha != 1:
                total_image /= tot_alpha
            final_image = float2int_image(total_image,
                                          target_dtype=image.dtype)
        elif needs_normalisation:
            window = self.get_channel_window(0)
            final_image = float2int_image(normalise_values(image, window['min'], window['max']),
                                          target_dtype=image.dtype)
        else:
            final_image = image
        return final_image

    def asarray(self, pixel_size: list = [], **slicing) -> np.ndarray:
        # expects x0, x1, y0, y1, ...
        x0, x1 = slicing.get('x0', 0), slicing.get('x1', -1)
        y0, y1 = slicing.get('y0', 0), slicing.get('y1', -1)
        # allow custom pixel size
        if pixel_size:
            level, factor, _ = get_best_level_factor(self, pixel_size)
            size0 = np.round(np.multiply(self.sizes[level], factor)).astype(int)
        else:
            level, factor = self.best_level, self.best_factor
            size0 = self.get_size()

        if x1 < 0 or y1 < 0:
            x1, y1 = size0
        if np.mean(factor) != 1:
            slicing['x0'], slicing['y0'] = np.round(np.divide([x0, y0], factor)).astype(int)
            slicing['x1'], slicing['y1'] = np.round(np.divide([x1, y1], factor)).astype(int)
        image0 = self._asarray_level(level=level, **slicing)
        if np.mean(factor) != 1:
            size1 = x1 - x0, y1 - y0
            image = image_resize(image0, size1, dimension_order=self.get_dimension_order())
        else:
            image = image0
        return image

    def asarray_um(self, **slicing):
        pixel_size = self.get_pixel_size_micrometer()[:2]
        slicing['x0'], slicing['y0'] = np.divide([slicing.get('x0'), slicing.get('y0')], pixel_size)
        slicing['x1'], slicing['y1'] = np.divide([slicing.get('x1'), slicing.get('y1')], pixel_size)
        return self.asarray(**slicing)

    def asdask(self, chunk_size: tuple) -> da.Array:
        chunk_shape = list(np.flip(chunk_size))
        while len(chunk_shape) < 3:
            chunk_shape = [1] + chunk_shape
        chunk_shape = [self.get_nchannels()] + chunk_shape
        while len(chunk_shape) < 5:
            chunk_shape = [1] + chunk_shape
        chunks = np.ceil(np.flip(self.get_size_xyzct()) / chunk_shape).astype(int)
        w, h = self.get_size()

        delayed_reader = dask.delayed(self.asarray)
        dtype = self.get_pixel_type()

        dask_times = []
        for ti in range(chunks[0]):
            dask_planes = []
            for zi in range(chunks[2]):
                dask_rows = []
                for yi in range(chunks[3]):
                    dask_row = []
                    for xi in range(chunks[4]):
                        shape = list(chunk_shape).copy()
                        x0, x1 = xi * shape[4], (xi + 1) * shape[4]
                        y0, y1 = yi * shape[3], (yi + 1) * shape[3]
                        if x1 > w:
                            x1 = w
                            shape[4] = w - x0
                        if y1 > h:
                            y1 = h
                            shape[3] = h - y0
                        z = zi * shape[2]
                        t = ti * shape[0]
                        dask_tile = da.from_delayed(delayed_reader(x0=x0, x1=x1, y0=y0, y1=y1, z=z, t=t),
                                                    shape=shape, dtype=dtype)
                        dask_row.append(dask_tile)
                    dask_rows.append(da.concatenate(dask_row, axis=4))
                dask_planes.append(da.concatenate(dask_rows, axis=3))
            dask_times.append(da.concatenate(dask_planes, axis=2))
        dask_data = da.concatenate(dask_times, axis=0)
        return dask_data

    def clone_empty(self) -> np.ndarray:
        return np.zeros(self.get_shape(), dtype=self.get_pixel_type())

    def produce_chunks(self, chunk_size: tuple) -> tuple:
        w, h = self.get_size()
        ny = int(np.ceil(h / chunk_size[1]))
        nx = int(np.ceil(w / chunk_size[0]))
        for chunky in range(ny):
            for chunkx in range(nx):
                x0, y0 = chunkx * chunk_size[0], chunky * chunk_size[1]
                x1, y1 = min((chunkx + 1) * chunk_size[0], w), min((chunky + 1) * chunk_size[1], h)
                indices = 0, 0, 0, y0, x0
                yield indices, self.asarray(x0=x0, x1=x1, y0=y0, y1=y1)

    def get_metadata(self) -> dict:
        return self.metadata

    #def create_xml_metadata(self, output_filename: str, combine_rgb: bool = True, pyramid_sizes_add: list = None) -> str:
    #    return create_ome_metadata(self, output_filename, combine_rgb=combine_rgb, pyramid_sizes_add=pyramid_sizes_add)

    def _find_metadata(self):
        raise NotImplementedError('Implement method in subclass')

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        raise NotImplementedError('Implement method in subclass')

    def close(self):
        pass


def get_resolution_from_pixel_size(pixel_size: list) -> tuple:
    conversions = {
        'cm': (1, 'centimeter'),
        'mm': (1, 'millimeter'),
        'µm': (1, 'micrometer'),
        'um': (1, 'micrometer'),
        'nm': (1000, 'micrometer'),
        'nanometer': (1000, 'micrometer'),
    }
    resolutions = []
    resolutions_unit = None
    if len(pixel_size) > 0:
        units = []
        for size, unit in pixel_size:
            if size != 0 and size != 1:
                resolution = 1 / size
                resolutions.append(resolution)
                if unit != '':
                    units.append(unit)
        if len(units) > 0:
            resolutions_unit = units[0]
            if resolutions_unit in conversions:
                conversion = conversions[resolutions_unit]
                resolutions = list(np.multiply(resolutions, conversion[0]))
                resolutions_unit = conversion[1]
    if len(resolutions) == 0:
        resolutions = None
    return resolutions, resolutions_unit


def get_best_level_factor(source: OmeSource, target_pixel_size: list) -> tuple:
    # find best pixel_size level and corresponding factor
    if source.source_pixel_size:
        target_factor = np.divide(get_value_units_micrometer(source.source_pixel_size)[:2],
                                  get_value_units_micrometer(target_pixel_size)[:2])
    else:
        target_factor = 1
    best_level = 0
    best_factor = None
    for level, size in enumerate(source.sizes):
        factor = np.divide(size, source.sizes[0])
        if (np.all(factor > target_factor)
                or np.all(np.isclose(factor, target_factor, rtol=1e-4))
                or best_factor is None):
            best_level = level
            best_factor = factor
    return best_level, target_factor / best_factor, target_factor


def get_best_size(sizes: list, target_size: tuple) -> tuple:
    # find largest scale but smaller to 1
    best_index = 0
    best_scale = 0
    for index, size in enumerate(sizes):
        scale = np.mean(np.divide(target_size, size))
        if 1 >= scale > best_scale:
            best_index = index
            best_scale = scale
    return sizes[best_index], best_index
