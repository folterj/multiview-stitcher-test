from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr, NgffMultiscales, \
    Omero, OmeroChannel, OmeroWindow
from ngff_zarr.v06.zarr_metadata import Metadata
import ome_zarr.format
import zarr

from muvis_align.image.color_conversion import rgba_to_hexrgb
from muvis_align.util import create_chunk_dict
from muvis_align.constants import default_ome_zarr_version, default_chunk_size
from muvis_align.image.util import create_compression_filter, build_missing_pyramid_levels
from muvis_align.image.ome_zarr_util import get_channel_window


def save_ome_zarr(filename, datas, dim_order, pixel_size, channels, translations, rotations, pyramid_downsample=None,
                  compression=None, ome_version=default_ome_zarr_version):
    # experimental: if list of images, store as series of images in ome-zarr format
    is_series = isinstance(datas, list)
    if not is_series:
        datas = [datas]

    zarr_format, ome_zarr_format = get_ome_zarr_format(ome_version)

    root = zarr.create_group(store=filename, zarr_format=zarr_format, overwrite=True)
    multi_metadata = []
    omero_metadata = None
    for index, data in enumerate(datas):
        translation = translations[index] if translations is not None else None
        rotation = rotations[index] if rotations is not None else None
        if is_series:
            path = filename + '/' + str(index)
        else:
            path = filename
        metadata = save_ome_image(data, path=path, dim_order=dim_order, pixel_size=pixel_size, channels=channels,
                                  translation=translation, rotation=rotation, pyramid_downsample=pyramid_downsample,
                                  compression=compression, ome_version=ome_version)

        if is_series:
            multi_metadata.append(metadata)
            if metadata:
                omero_metadata = metadata.omero

    if is_series:
        root.attrs['multiscales'] = multi_metadata
        root.attrs['omero'] = omero_metadata


def save_ome_image(data, path, dim_order, pixel_size, channels, translation, rotation,
                   pyramid_downsample=None, compression=None, ome_version=default_ome_zarr_version):

    storage_options = {}
    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    axes_units = {dim: 'micrometer' for dim in dim_order if dim in 'xyz'}
    image = to_ngff_image(data, dims=dim_order, scale=pixel_size, translation=translation, axes_units=axes_units)
    multiscales = to_multiscales(image, scale_factors=pyramid_downsample, chunks=create_chunk_dict(default_chunk_size, dim_order))

    if channels:
        omero = Omero(channels=[OmeroChannel(label=channel.get('label', f'Channel {index}'),
                                             color=rgba_to_hexrgb(channel.get('color')),
                                             window=OmeroWindow(**get_channel_window(multiscales.images[-1].data, dim_order, index)))
                                for index, channel in enumerate(channels)])

        multiscales.metadata.omero = omero

    chunks_per_shard = None

    to_ngff_zarr(path, multiscales, chunks_per_shard=chunks_per_shard, version=ome_version, **storage_options)

    return multiscales.metadata


def save_ome_multiscale_levels(path, levels, dim_order, channels, translation,
                               pyramid_downsample=2, min_size=default_chunk_size,
                               compression=None, ome_version=default_ome_zarr_version):
    """Write pre-built pyramid levels (e.g. a source's own native resolutions) exactly as
    given - no resampling - unlike save_ome_image()/to_multiscales(), which always derives
    every level but the first from one input via resampling. Only pads with additional coarser
    levels, resampled from the smallest given level, if fewer than build_missing_pyramid_levels()'s
    own stopping criterion (shrunk to at most min_size) would otherwise leave.

    levels: list of (data, pixel_size) pairs, finest first - data can be numpy or dask.
    """
    storage_options = {}
    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    datas = [data for data, _ in levels]
    pixel_sizes = [pixel_size for _, pixel_size in levels]
    # build_missing_pyramid_levels() already implements "keep halving until <= min_size" -
    # reused here from the smallest level already given, so it adds nothing when that's
    # already small enough on its own
    extra_datas, extra_pixel_sizes = build_missing_pyramid_levels(
        datas[-1], dim_order, pixel_sizes[-1], pyramid_downsample=pyramid_downsample, min_size=min_size)
    datas += extra_datas[1:]
    pixel_sizes += extra_pixel_sizes[1:]

    axes_units = {dim: 'micrometer' for dim in dim_order if dim in 'xyz'}
    chunks = create_chunk_dict(default_chunk_size, dim_order)
    images = []
    datasets = []
    coordinate_systems = None
    for index, (data, pixel_size) in enumerate(zip(datas, pixel_sizes)):
        # to_multiscales(scale_factors=[]) just wraps one already-built level with correct OME
        # metadata (axes/coordinateSystems/scale/translation) - no resampling happens for it
        ngff_image = to_ngff_image(data, dims=dim_order, scale=pixel_size, translation=translation,
                                   axes_units=axes_units)
        level_multiscales = to_multiscales(ngff_image, scale_factors=[], chunks=chunks)
        images.append(level_multiscales.images[0])
        dataset = level_multiscales.metadata.datasets[0]
        dataset.path = f'scale{index}/image'
        datasets.append(dataset)
        if coordinate_systems is None:
            coordinate_systems = level_multiscales.metadata.coordinateSystems

    metadata = Metadata(coordinateSystems=coordinate_systems, datasets=datasets)
    if channels:
        omero = Omero(channels=[OmeroChannel(label=channel.get('label', f'Channel {index}'),
                                             color=rgba_to_hexrgb(channel.get('color')),
                                             window=OmeroWindow(**get_channel_window(images[-1].data, dim_order, index)))
                                for index, channel in enumerate(channels)])
        metadata.omero = omero

    multiscales = NgffMultiscales(images=images, metadata=metadata)
    to_ngff_zarr(path, multiscales, version=ome_version, **storage_options)

    return metadata


def get_ome_zarr_format(ome_version):
    if str(ome_version) == '0.4':
        ome_zarr_format = ome_zarr.format.FormatV04()
    elif str(ome_version) == '0.5':
        ome_zarr_format = ome_zarr.format.FormatV05()
    else:
        ome_zarr_format = ome_zarr.format.CurrentFormat()
    zarr_format = 3 if float(ome_zarr_format.version) >= 0.5 else 2
    return zarr_format, ome_zarr_format
