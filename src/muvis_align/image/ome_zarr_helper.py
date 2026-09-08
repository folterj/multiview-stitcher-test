from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr, NgffMultiscales, \
    Omero, OmeroChannel, OmeroWindow
from ngff_zarr.v06.zarr_metadata import Metadata
import ome_zarr.format
import zarr

from muvis_align.image.color_conversion import rgba_to_hexrgb
from muvis_align.util import create_chunk_dict
from muvis_align.constants import default_ome_zarr_version, default_chunk_size
from muvis_align.image.util import calc_pyramid_level_factors, create_compression_filter
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


def get_padding_scale_factors(shape, dim_order, **kwargs):
    """Cumulative per-level downsample factors needed to extend a pyramid whose coarsest level
    is `shape` down to one small enough to draw a zoomed-out overview from - calc_pyramid_level_
    factors()' rule, the same one build_missing_pyramid_levels() applies when synthesizing those
    levels on the reader side. Writing them means a store carries the levels a reader would
    otherwise have to invent - and, for a chunked store, could not invent cheaply: strided
    subsampling of a zarr array still reads every chunk it touches, unlike the already-decoded
    single page of a non-pyramidal TIFF.

    Already in ngff_zarr's own form: one dict per extra level, giving that level's factor
    relative to the input, cumulative rather than per-step.
    """
    return calc_pyramid_level_factors(
        {dim: size for dim, size in zip(dim_order, shape) if dim in 'xyz'}, **kwargs)


def save_ome_multiscale_levels(path, levels, dim_order, channels, translation,
                               min_length=128, compression=None, ome_version=default_ome_zarr_version):
    """Write pre-built pyramid levels (e.g. a source's own native resolutions) exactly as
    given - no resampling - unlike save_ome_image()/to_multiscales(), which always derives
    every level but the first from one input via resampling.

    Pads with additional coarser levels beyond the smallest given one, down to the same
    "small enough to draw a zoomed-out overview from" threshold the reader side uses
    (build_missing_pyramid_levels' min_size, i.e. default_chunk_size). Those extra levels are
    genuinely resampled (via ngff_zarr's own default downsampling method) since nothing at that
    resolution exists in the source to preserve.

    The per-level factors are computed here rather than delegated to to_multiscales(
    scale_factors=<int>): that int form calls ngff_zarr's private _ngff_image_scale_factors(),
    which stops as soon as every spatial dim is below *twice* the chunk size and never emits a
    level smaller than one chunk. For a 1024 chunk that leaves the coarsest written level at
    >=1024px, a level coarser than which the reader would have synthesized for itself from a
    non-pyramidal source - so a converted OME-Zarr ended up with a coarser-level gap exactly
    where a zoomed-out preview needs one, and reading it back selected a much finer level than
    the requested preview scale (fusing, and holding in memory, far more than asked for).

    levels: list of (data, pixel_size) pairs, finest first - data can be numpy or dask.
    """
    storage_options = {}
    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    axes_units = {dim: 'micrometer' for dim in dim_order if dim in 'xyz'}
    chunks = create_chunk_dict(default_chunk_size, dim_order)
    images = []
    datasets = []
    coordinate_systems = None
    for data, pixel_size in levels:
        # to_multiscales(scale_factors=[]) just wraps one already-built level with correct OME
        # metadata (axes/coordinateSystems/scale/translation) - no resampling happens for it
        ngff_image = to_ngff_image(data, dims=dim_order, scale=pixel_size, translation=translation,
                                   axes_units=axes_units)
        level_multiscales = to_multiscales(ngff_image, scale_factors=[], chunks=chunks)
        images.append(level_multiscales.images[0])
        datasets.append(level_multiscales.metadata.datasets[0])
        if coordinate_systems is None:
            coordinate_systems = level_multiscales.metadata.coordinateSystems

    smallest_data, smallest_pixel_size = levels[-1]
    smallest_ngff_image = to_ngff_image(smallest_data, dims=dim_order, scale=smallest_pixel_size,
                                        translation=translation, axes_units=axes_units)
    extra_scale_factors = get_padding_scale_factors(smallest_data.shape, dim_order)
    if extra_scale_factors:
        extra_multiscales = to_multiscales(smallest_ngff_image, scale_factors=extra_scale_factors,
                                           chunks=chunks)
        images += extra_multiscales.images[1:]
        datasets += extra_multiscales.metadata.datasets[1:]

    # renumber every dataset path sequentially - each was independently built starting from
    # 'scale0/image', so native and padding levels alike would otherwise collide/repeat
    for index, dataset in enumerate(datasets):
        dataset.path = f'scale{index}/image'

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
