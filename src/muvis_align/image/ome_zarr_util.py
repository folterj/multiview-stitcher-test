import json
import os

import numpy as np
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.color_conversion import *
from muvis_align.image.util import get_image_quantile


def create_axes_metadata(dimension_order):
    axes = []
    for dimension in dimension_order:
        unit1 = None
        if dimension == 't':
            type1 = 'time'
            unit1 = 'millisecond'
        elif dimension == 'c':
            type1 = 'channel'
        else:
            type1 = 'space'
            unit1 = 'micrometer'
        axis = {'name': dimension, 'type': type1}
        if unit1 is not None and unit1 != '':
            axis['unit'] = unit1
        axes.append(axis)
    return axes


def create_transformation_metadata(dimension_order, pixel_size_um, factor, translation_um={}, rotation=None):
    metadata = []
    metadata_scale = []
    metadata_translation = []
    for dim in dimension_order:
        if dim in 'xy':
            pixel_size_scale1 = pixel_size_um[dim] * factor
        else:
            pixel_size_scale1 = 1
        if pixel_size_scale1 == 0:
            pixel_size_scale1 = 1
        metadata_scale.append(pixel_size_scale1)

        # translation_pyramid = translation + (scale - 1) * pixel_size / 2
        if dim in 'xy':
            correction = (factor - 1) * pixel_size_um[dim] / 2
        else:
            correction = 0
        metadata_translation.append(translation_um.get(dim, 0) + correction)

    metadata.append({'type': 'scale', 'scale': metadata_scale})
    metadata.append({'type': 'translation', 'translation': metadata_translation})
    # Supported in ome-zarr V0.6
    #if rotation is not None:
    #    metadata.append({'type': 'rotation', 'rotation': rotation})
    return metadata


def create_channel_metadata(source, ome_version):
    channels = source.get_channels()
    nchannels = source.get_nchannels()

    if len(channels) < nchannels == 3:
        labels = ['Red', 'Green', 'Blue']
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        channels = [{'label': label, 'color': color} for label, color in zip(labels, colors)]

    omezarr_channels = []
    for channeli, channel0 in enumerate(channels):
        channel = channel0.copy()
        color = channel.get('color', (1, 1, 1, 1))
        channel['color'] = rgba_to_hexrgb(color)
        if 'window' not in channel:
            channel['window'] = source.get_channel_window(channeli)
        omezarr_channels.append(channel)

    metadata = {
        'version': ome_version,
        'channels': omezarr_channels,
    }
    return metadata


def create_channel_ome_metadata(data, dimension_order, channels, ome_version):
    if 'c' in dimension_order:
        nchannels = data.shape[dimension_order.index('c')]
    else:
        nchannels = 1
    if channels is None or len(channels) < nchannels:
        if nchannels == 3:
            labels = ['Red', 'Green', 'Blue']
            colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
            channels = [{'label': label, 'color': color} for label, color in zip(labels, colors)]
        else:
            channels = [{'label': f'Channel {channeli}'} for channeli in range(nchannels)]

    omezarr_channels = []
    for channeli, channel0 in enumerate(channels):
        channel = channel0.copy()
        color = channel.get('color', (1, 1, 1, 1))
        channel['color'] = rgba_to_hexrgb(color)
        if 'window' not in channel:
            channel['window'] = get_channel_window(data, dimension_order, channeli)
        omezarr_channels.append(channel)

    metadata = {
        'version': ome_version,
        'channels': omezarr_channels,
    }
    return metadata


def get_channel_window(data, dimension_order, channeli):
    min_quantile = 0.001
    max_quantile = 0.999

    if data.dtype.kind == 'f':
        #info = np.finfo(dtype)
        min, max = 0, 1
    else:
        info = np.iinfo(data.dtype)
        min, max = info.min, info.max

    if data.dtype.itemsize == 1:
        start, end = min, max
    else:
        if 'c' in dimension_order:
            data = np.take(data, channeli, axis=dimension_order.index('c'))
        start, end = get_image_quantile(data, min_quantile), get_image_quantile(data, max_quantile)
    window = {'min': min, 'max': max, 'start': start, 'end': end}
    return window


def scale_dimensions_xy(shape0, dimension_order, scale):
    shape = []
    if scale == 1:
        return shape0
    for shape1, dimension in zip(shape0, dimension_order):
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape.append(shape1)
    return shape


def scale_dimensions_dict(shape0, scale):
    shape = {}
    if scale == 1:
        return shape0
    for dimension, shape1 in shape0.items():
        if dimension[0] in ['x', 'y']:
            shape1 = int(shape1 * scale)
        shape[dimension] = shape1
    return shape


def ngff_dims_to_sim_dims(file_dims):
    """The dims a sim built from `file_dims` ends up with: si_utils.SPATIAL_IMAGE_DIMS order,
    keeping whichever spatial dims the file has and always including 't' and 'c' - which
    si_utils.get_sim_from_array forces to exist at size 1 when the file has none (the same
    multiview_stitcher convention util.ensure_spatial_image_dims matches). Reproducing the rule
    here is what lets source metadata be read without building the msim (a DataTree of one sim
    per level) just to ask it for its own shapes.
    """
    return [dim for dim in si_utils.SPATIAL_IMAGE_DIMS
            if dim in sim_forced_dims or dim in file_dims]


sim_forced_dims = ('t', 'c')
sim_spatial_dims = tuple(dim for dim in si_utils.SPATIAL_IMAGE_DIMS if dim not in sim_forced_dims)


def read_ome_zarr_source_metadata(path):
    """Every piece of metadata an ImageSource needs from an OME-Zarr - per-level shapes and
    pixel sizes, dtype, origin, channel count, omero - *without* constructing a msim.

    Building the msim eagerly (read_msim_from_ome_zarr) costs one zarr.json read per pyramid
    level plus a full xarray DataTree of one sim per level, measured at ~100ms per source: for
    a few thousand sources that is minutes of project load spent on data nothing has asked for
    yet. A v0.5/zarr-v3 store already carries every level's shape and dtype in its consolidated
    root zarr.json, so the whole thing is one read (~1ms); ngff_zarr does not use that
    consolidated metadata (it hands the *store*, not the opened group, to its own
    Metadata._from_zarr_attrs, which reopens each level by path), hence reading it directly.

    Falls back to ngff_zarr's own (version-aware) metadata parse for anything the fast path
    cannot fully account for - a non-consolidated store, v0.4, a remote URL, or any coordinate
    transformation beyond the plain per-dataset scale/translation pair. That fallback still
    skips the DataTree construction, so it is cheaper than building the msim either way.

    Returns a dict of dimension_order, shapes, dtype, pixel_sizes, position, nchannels and
    omero - all already expressed in the forced t/c sim dimension order (see above), so a
    caller can assign them straight onto an ImageSource.
    """
    metadata = _read_consolidated_ome_zarr_metadata(path)
    if metadata is None:
        metadata = _read_ngff_ome_zarr_metadata(path)
    return metadata


def _build_source_metadata(file_dims, file_shapes, dtype, scales, translation, omero, paths=None):
    """Shared tail of both read paths: map per-level file dims/shapes/scales onto the forced
    t/c sim dimension order. scales is one dict per level (keyed by file dim), translation one
    dict for the finest level; either may omit dims (defaulting to 1.0 / 0.0).

    `paths` are the levels' own array paths within the store, and `file_shapes` their on-disk
    shapes - both as read, before the t/c mapping - so a caller can open each level's array
    directly (see ZarrImageSource._load_data) instead of going back through a full msim read.
    """
    sim_dims = ngff_dims_to_sim_dims(file_dims)
    spatial_dims = [dim for dim in sim_dims if dim in sim_spatial_dims]
    shapes = []
    for shape in file_shapes:
        sizes = dict(zip(file_dims, shape))
        shapes.append(tuple(sizes.get(dim, 1) for dim in sim_dims))
    pixel_sizes = [{dim: float(scale.get(dim, 1.0)) for dim in spatial_dims} for scale in scales]
    position = {dim: float(translation.get(dim, 0.0)) for dim in spatial_dims}
    nchannels = dict(zip(sim_dims, shapes[0])).get('c', 1)
    return {'dimension_order': ''.join(sim_dims), 'shapes': shapes, 'dtype': np.dtype(dtype),
            'pixel_sizes': pixel_sizes, 'position': position, 'nchannels': int(nchannels),
            'omero': omero, 'paths': paths, 'file_shapes': list(file_shapes)}


def _read_consolidated_ome_zarr_metadata(path):
    """Fast path: parse the consolidated root zarr.json directly. Returns None (rather than
    raising or guessing) for anything it cannot fully account for, leaving the caller to fall
    back to ngff_zarr's own parse - the point of this path is to be exactly equivalent wherever
    it applies, never to be approximately right more often.
    """
    root_file = os.path.join(str(path), 'zarr.json')
    if not os.path.isfile(root_file):
        return None
    try:
        with open(root_file) as file:
            root = json.load(file)
    except (OSError, ValueError):
        return None
    if root.get('zarr_format') != 3:
        return None
    nodes = root.get('consolidated_metadata', {}).get('metadata')
    if not nodes:
        return None
    multiscales = root.get('attributes', {}).get('ome', {}).get('multiscales')
    if not multiscales:
        return None
    multiscale = multiscales[0]
    # a multiscales-level transform composes with the per-dataset ones - not handled here
    if multiscale.get('coordinateTransformations'):
        return None
    file_dims = tuple(axis['name'] for axis in multiscale.get('axes', []))
    known_dims = si_utils.SPATIAL_IMAGE_DIMS
    if not file_dims or any(dim not in known_dims for dim in file_dims):
        return None

    file_shapes, scales, dtypes = [], [], set()
    translation = {}
    for index, dataset in enumerate(multiscale.get('datasets', [])):
        node = nodes.get(dataset.get('path'))
        if not node or node.get('node_type') != 'array':
            return None
        transforms = dataset.get('coordinateTransformations', [])
        if any(transform.get('type') not in ('scale', 'translation') for transform in transforms):
            return None
        scale_values = next((transform['scale'] for transform in transforms
                             if transform.get('type') == 'scale'), None)
        shape = node.get('shape')
        if scale_values is None or len(scale_values) != len(file_dims):
            return None
        if shape is None or len(shape) != len(file_dims):
            return None
        file_shapes.append(tuple(shape))
        scales.append(dict(zip(file_dims, scale_values)))
        dtypes.add(node.get('data_type'))
        if index == 0:
            translation_values = next((transform['translation'] for transform in transforms
                                       if transform.get('type') == 'translation'), None)
            if translation_values is not None:
                if len(translation_values) != len(file_dims):
                    return None
                translation = dict(zip(file_dims, translation_values))
    if not file_shapes or len(dtypes) != 1:
        return None
    try:
        dtype = np.dtype(dtypes.pop())
    except TypeError:
        return None

    ome = root.get('attributes', {}).get('ome', {})
    paths = [dataset.get('path') for dataset in multiscale.get('datasets', [])]
    return _build_source_metadata(file_dims, file_shapes, dtype, scales, translation,
                                  ome.get('omero'), paths=paths)


def _read_ngff_ome_zarr_metadata(path):
    """Fallback: ngff_zarr's own version-aware parse (v0.4, non-consolidated, remote stores),
    still without building the msim - only .dims/.data.shape/.scale/.translation are read off
    each level's NgffImage, no sim or DataTree is constructed.
    """
    # imported here rather than at module scope: multiview_stitcher.ngff_utils pulls in a large
    # dependency chain that the fast path above never needs
    from multiview_stitcher import ngff_utils

    multiscales = ngff_utils.read_ngff_multiscales(path)
    images = multiscales.images
    file_dims = tuple(images[0].dims)
    omero = getattr(multiscales.metadata, 'omero', None)
    if omero is not None and not isinstance(omero, dict):
        dump = getattr(omero, 'model_dump', None)
        omero = dump() if dump is not None else None
    datasets = getattr(multiscales.metadata, 'datasets', None) or []
    paths = [getattr(dataset, 'path', None) for dataset in datasets]
    if len(paths) != len(images) or not all(paths):
        # no usable per-level array paths - _load_data() then falls back to the msim reader
        paths = None
    return _build_source_metadata(
        file_dims,
        [tuple(image.data.shape) for image in images],
        images[0].data.dtype,
        [dict(image.scale) for image in images],
        dict(images[0].translation),
        omero,
        paths=paths)
