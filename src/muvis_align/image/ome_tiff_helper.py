from xml.etree import ElementTree

import dask.array as da
import zarr
from ome_zarr.scale import Scaler
from tifffile import TiffWriter, tifffile

try:
    # ngff_zarr's own tifffile-axes -> NGFF-dims mapping (and the array reshape that goes with
    # it), reused rather than reimplemented so a fast metadata or array read can never disagree
    # with what ngff_zarr itself would build from the same file. Private, hence guarded: if they
    # move, read_tiff_source_metadata()/read_tiff_level_arrays() decline and the caller falls
    # back to ngff_zarr itself.
    from ngff_zarr.tiff_to_ngff_image import _map_tiff_axes_to_ngff as map_tiff_axes_to_ngff
    from ngff_zarr.tiff_to_ngff_image import _normalize_unit as normalize_ome_unit
    from ngff_zarr.tiff_to_ngff_image import _reshape_tiff_for_channels as reshape_tiff_for_channels
except ImportError:  # pragma: no cover - depends on the installed ngff_zarr
    map_tiff_axes_to_ngff = None
    reshape_tiff_for_channels = None

    def normalize_ome_unit(unit):
        return unit

from muvis_align.constants import default_chunk_size
from muvis_align.image.color_conversion import hexrgb_to_rgba, rgba_to_int
from muvis_align.util import *


def load_tiff(filename):
    return tifffile.imread(filename)


def extract_ome_translation(filename):
    with tifffile.TiffFile(filename) as tif:
        # .is_ome/.ome_metadata only read the description tag - deliberately not .series, which
        # for an OME-TIFF parses the whole OME XML a second time to build its series/levels
        if not tif.is_ome or tif.ome_metadata is None:
            return {}
        return extract_ome_translation_from_xml(tif.ome_metadata)


def extract_ome_translation_from_xml(ome_xml, **kwargs):
    """The first Image's first Plane position, in um - see extract_ome_image_metadata()."""
    return extract_ome_image_metadata(ome_xml, **kwargs)['position']


def extract_ome_image_metadata(ome_xml, chunk_size=64 * 1024):
    """The first Image's geometry and channels: position (um), physical pixel size and its
    units, and channel names/colours - everything a source reads out of an OME-TIFF's XML,
    gathered in one pass.

    Parsed incrementally, stopping as soon as the answer is settled, instead of building a DOM
    of the whole document: a multi-file OME-TIFF repeats the *entire dataset's* OME XML in every
    file's header, so for a 4733-file set that is a ~2.4MB XML per file, and a full parse of it
    measured ~210ms - per file, i.e. ~16 CPU-minutes across the set to read a handful of
    attributes. Everything wanted here lives in the first <Image>, and the parse stops at the
    second one, so the cost does not depend on the size of the dataset at all. (Fed in chunks
    rather than via a StringIO over the whole string, which alone costs a full copy of it -
    measured 3.7ms of 4.1ms at 2.4MB, dwarfing the ~0.4ms parse.)

    `position` is {} when the XML describes more than one Image, which preserves the behaviour
    of the xml2dict implementation this replaces: repeated <Image> elements became a list, which
    its 'Pixels' in metadata['Image'] test failed on, so such a file has always yielded no
    position (positions then come from source_metadata instead). Reading it properly would mean
    matching the Image whose TiffData/UUID FileName is this file rather than taking the first -
    a behaviour change, not a refactor, so it is left alone here. `scale`/`units`/channels are
    taken from the first Image either way, matching what ngff_zarr does (it indexes Image by
    series, and a source only ever reads series 0).
    """
    parser = ElementTree.XMLPullParser(events=['start'])
    position, scale, units = {}, {}, {}
    channel_names, channel_colors = [], []
    images = 0
    for start in range(0, len(ome_xml), chunk_size):
        parser.feed(ome_xml[start:start + chunk_size])
        for _event, element in parser.read_events():
            # tags carry the OME namespace, e.g. '{http://...}Plane'
            tag = element.tag.rpartition('}')[2]
            if tag == 'Image':
                images += 1
                if images > 1:
                    # a second Image settles it - nothing further belongs to this file's own
                    return {'position': {}, 'scale': scale, 'units': units,
                            'channel_names': channel_names, 'channel_colors': channel_colors}
            elif tag == 'Pixels':
                for dim in 'XYZ':
                    value = element.get(f'PhysicalSize{dim}')
                    if value is not None:
                        try:
                            scale[dim.lower()] = float(value)
                        except ValueError:
                            continue
                        unit = normalize_ome_unit(element.get(f'PhysicalSize{dim}Unit'))
                        if unit is not None:
                            units[dim.lower()] = unit
            elif tag == 'Channel':
                channel_names.append(element.get('Name', ''))
                channel_colors.append(element.get('Color'))
            elif tag == 'Plane' and not position:
                for dim in ['X', 'Y', 'Z']:
                    key = f'Position{dim}'
                    value = element.get(key)
                    if value is not None:
                        position[dim.lower()] = convert_to_um(float(value),
                                                              element.get(f'{key}Unit', 'um'))
    return {'position': position, 'scale': scale, 'units': units,
            'channel_names': channel_names, 'channel_colors': channel_colors}


def read_tiff_source_metadata(filename):
    """Every piece of metadata an ImageSource needs from a TIFF - per-level shapes and pixel
    sizes, dtype, dimension order, origin, channels - read straight off tifffile, with no zarr
    store and no dask array built. Returns None if it cannot be done faithfully, leaving the
    caller on ngff_zarr's own (array-building) path.

    ngff_zarr.tiff_file_to_ngff_images() is the reference for all of this, but obtaining it from
    there costs ~8ms per source: it opens tif.aszarr(), walks the zarr group per pyramid level
    and wraps each in a dask array - ~18 round-trips through zarr's async/sync bridge - none of
    which project load needs, since only metadata is read until something asks for pixels. For
    an OME-TIFF it additionally DOM-parses the whole OME XML (findall('.//ome:Image')), the same
    per-file cost over the whole dataset's XML described in extract_ome_image_metadata().
    Measured on a real pyramidal tile: 0.68ms here against 8.41ms there.

    The tifffile-axes-to-NGFF-dims mapping is ngff_zarr's own _map_tiff_axes_to_ngff (channel-
    like axes flattened - 'S' samples included, so RGB lands on 'c' - and unsupported axes
    dropped): deliberately reused rather than reimplemented, since a divergence there would
    silently mis-order dimensions.
    """
    if map_tiff_axes_to_ngff is None:
        return None
    with tifffile.TiffFile(filename) as tif:
        series = tif.series[0]
        axes = series.axes
        if not axes:
            return None
        level_shapes = [level.shape for level in series.levels]
        dtype = series.dtype
        # .is_ome/.ome_metadata read only the description tag, not the series structure
        ome = (extract_ome_image_metadata(tif.ome_metadata)
               if tif.is_ome and tif.ome_metadata is not None else None)

    dims, _, _, _ = map_tiff_axes_to_ngff(axes, series.shape)
    shapes = [map_tiff_axes_to_ngff(axes, shape)[1] for shape in level_shapes]
    spatial_dims = [dim for dim in dims if dim in 'xyz']
    if not shapes or not spatial_dims:
        return None

    # ngff_zarr takes the physical pixel size from OME PhysicalSize*, defaulting any spatial dim
    # the XML does not give to 1.0 - and ignores the TIFF resolution tags entirely, so a
    # non-OME TIFF is simply 1.0 per dim
    ome_scale = (ome or {}).get('scale') or {}
    ome_units = (ome or {}).get('units') or {}
    axis_of = {dim: index for index, dim in enumerate(dims)}
    pixel_sizes = []
    for shape in shapes:
        # per level, ngff_zarr scales the base value by the realised extent ratio rather than by
        # the nominal downsample factor
        pixel_sizes.append({
            dim: convert_to_um(ome_scale.get(dim, 1.0) * shapes[0][axis_of[dim]] / shape[axis_of[dim]],
                               ome_units.get(dim, 'um'))
            for dim in spatial_dims})

    channels = []
    for name, color in zip((ome or {}).get('channel_names') or [],
                           (ome or {}).get('channel_colors') or []):
        channel = {'label': name}
        if color:
            channel['color'] = hexrgb_to_rgba(color)
        channels.append(channel)

    return {'dimension_order': ''.join(dims), 'shapes': shapes, 'dtype': dtype,
            'pixel_sizes': pixel_sizes, 'position': (ome or {}).get('position') or {},
            'channels': channels}


def read_tiff_level_arrays(filename):
    """One dask array per pyramid level of the file's first series, opened straight off
    tifffile's own zarr store. Returns None if it cannot be done faithfully, leaving the caller
    on ngff_zarr's own path.

    The arrays are the same ones ngff_zarr.tiff_file_to_ngff_images() produces - it reaches them
    exactly this way (tif.aszarr() -> zarr group -> da.from_zarr per level, then its axis
    mapping and channel reshape, both reused here) - but without also rebuilding the metadata
    the source has already read for itself off tifffile, which is what the rest of that call
    costs. Measured per source: 3.2ms here against 4.7ms for a plain 800x800 tile, 6.6ms against
    9.8ms for a 4096x4096 4-level pyramid.

    The dask wrapping is deliberate, even though reading pixels off the zarr arrays directly is
    3-4x faster (37ms vs 129ms for a full 4096x4096 level, 75ms vs 190ms compressed): the whole
    pipeline downstream is lazy, and a zarr array is not. Slicing one reads immediately, which
    would turn build_missing_pyramid_levels()' near-free strided subsampling into a full decode
    at source load, and .data on a zarr-backed sim hands back a materialised numpy array rather
    than a graph. Handing zarr arrays to multiview_stitcher (which does support them natively)
    is the larger win, but it is a change to how the whole pipeline reads pixels, not to how a
    TIFF is opened.
    """
    if map_tiff_axes_to_ngff is None or reshape_tiff_for_channels is None:
        return None
    with tifffile.TiffFile(filename) as tif:
        series = tif.series[0]
        axes = series.axes
        if not axes:
            return None
        # multiscales=True so a single-level file yields a group too, and the levels come back
        # in the multiscales metadata's own order (highest resolution first) rather than in the
        # group's arbitrary iteration order
        group = zarr.open_group(store=tif.aszarr(series=0, multiscales=True), mode='r')
        attributes = group.attrs.get('ome', group.attrs)
        try:
            datasets = attributes['multiscales'][0]['datasets']
        except (KeyError, IndexError, TypeError):
            return None
        arrays = [group[dataset['path']] for dataset in datasets]
        # tif is closed on the way out: tifffile's store reopens the file itself whenever a
        # chunk is actually read, so holding a file handle open per source (thousands of them in
        # a project) buys nothing
        dims, _, channel_indices, dropped_indices = map_tiff_axes_to_ngff(axes, series.shape)

    datas = []
    for array in arrays:
        # the level's shape in NGFF terms (unsupported axes dropped, channel-like axes - 'S'
        # samples included - flattened onto 'c'), then the array reshaped to match it
        shape = map_tiff_axes_to_ngff(axes, array.shape)[1]
        data = da.from_zarr(array)
        if tuple(data.shape) != tuple(shape):
            data = reshape_tiff_for_channels(data, axes, tuple(array.shape), dims, tuple(shape),
                                             list(channel_indices), list(dropped_indices))
        if tuple(data.shape) != tuple(shape):
            # the reshape could not get there - decline rather than hand back levels whose
            # shapes disagree with the metadata read from the same file
            return None
        datas.append(data)
    return datas


def save_tiff(filename, data, dimension_order=None, pixel_size=None, tile_size=(default_chunk_size, default_chunk_size),
              compression='LZW'):
    _, resolution, resolution_unit = create_tiff_metadata(pixel_size, dimension_order)
    tifffile.imwrite(filename, data, tile=tile_size, compression=compression,
                     resolution=resolution, resolutionunit=resolution_unit)


def save_ome_tiff(filename, data, dimension_order, pixel_size, channels=[], positions=[], rotation=None,
                  tile_size=None, compression=None, pyramid_downsample=2, npyramid_add=None):

    ome_metadata, resolution0, resolution_unit0 = create_tiff_metadata(pixel_size, dimension_order,
                                                                       channels, positions, rotation, is_ome=True)
    # maximum size (w/o compression)
    max_size = data.size * data.itemsize
    size = max_size
    if pyramid_downsample is not None:
        scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)
        npyramid_add = scaler.max_layer
        for i in range(npyramid_add):
            size //= (scaler.downscale ** 2)
            max_size += size
    bigtiff = (max_size > 2 ** 32)

    if tile_size:
        tile_size = tile_size[-2:]  # assume order zyx (inversed xyz)
        shape_yx = [data.shape[dimension_order.index(dim)] for dim in 'yx']
        if np.any(np.array(tile_size) > np.array(shape_yx)):
            tile_size = None

    with TiffWriter(filename, bigtiff=bigtiff) as writer:
        for i in range(npyramid_add + 1):
            if i == 0:
                subifds = npyramid_add
                subfiletype = None
                metadata = ome_metadata
                resolution = resolution0[:2]
                resolutionunit = resolution_unit0
            else:
                subifds = None
                subfiletype = 1
                metadata = None
                resolution = None
                resolutionunit = None
                data = scaler.resize_image(data)
                data.rechunk()
            writer.write(data, subifds=subifds, subfiletype=subfiletype,
                         tile=tile_size, compression=compression,
                         resolution=resolution, resolutionunit=resolutionunit, metadata=metadata)


def create_tiff_metadata(pixel_size, dimension_order=None, channels=[], positions=[], rotation=None, is_ome=False):
    ome_metadata = None
    resolution = None
    resolution_unit = None

    if pixel_size is not None:
        resolution_unit = 'CENTIMETER'
        resolution = [1e4 / size for size in dict_to_xyz(pixel_size, 'xy')]

    if is_ome:
        ome_metadata = {'Creator': 'muvis-align'}
        if dimension_order is not None:
            #ome_metadata['DimensionOrder'] = dimension_order[::-1].upper()
            ome_metadata['axes'] = dimension_order.upper()
        ome_channels = []
        if pixel_size is not None:
            ome_metadata['PhysicalSizeX'] = pixel_size['x']
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeY'] = pixel_size['y']
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
            if 'z' in pixel_size:
                ome_metadata['PhysicalSizeZ'] = pixel_size['z']
                ome_metadata['PhysicalSizeZUnit'] = 'µm'
        if positions is not None and len(positions) > 0:
            plane_metadata = {}
            plane_metadata['PositionX'] = [position['x'] for position in positions]
            plane_metadata['PositionXUnit'] = ['µm' for _ in positions]
            plane_metadata['PositionY'] = [position['y'] for position in positions]
            plane_metadata['PositionYUnit'] = ['µm' for _ in positions]
            if 'z' in positions[0]:
                plane_metadata['PositionZ'] = [position['z'] for position in positions]
                plane_metadata['PositionZUnit'] = ['µm' for _ in positions]
            ome_metadata['Plane'] = plane_metadata
        if rotation is not None:
            ome_metadata['StructuredAnnotations'] = {'CommentAnnotation': {'Value': f'Angle: {rotation} degrees'}}
        for channeli, channel in enumerate(channels):
            ome_channel = {'Name': channel.get('label', str(channeli))}
            if 'color' in channel:
                ome_channel['Color'] = rgba_to_int(channel['color'])
            ome_channels.append(ome_channel)
        if ome_channels:
            ome_metadata['Channel'] = ome_channels
    return ome_metadata, resolution, resolution_unit
