import zarr
import ome_zarr.format
from ome_zarr.writer import write_image

from src.image.util import create_compression_filter, redimension_data
from src.image.ome_zarr_util import create_axes_metadata, create_transformation_metadata, create_channel_ome_metadata


def save_ome_zarr(filename, data, dimension_order, pixel_size, channels, translation, rotation,
                  tile_size=(256, 256), compression=None,
                  scaler=None, zarr_version=2, ome_version='0.4'):

    storage_options = {'dimension_separator': '/', 'chunks': tile_size}

    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    if 'z' not in dimension_order:
        # add Z dimension to be able to store Z position
        new_dimension_order = dimension_order.replace('yx', 'zyx')
        data = redimension_data(data, dimension_order, new_dimension_order)
        dimension_order = new_dimension_order

    axes = create_axes_metadata(dimension_order)

    if scaler is not None:
        npyramid_add = scaler.max_layer
        pyramid_downsample = scaler.downscale
    else:
        npyramid_add = 0
        pyramid_downsample = 1

    coordinate_transformations = []
    scale = 1
    for i in range(npyramid_add + 1):
        transform = create_transformation_metadata(dimension_order, pixel_size, scale, translation, rotation)
        coordinate_transformations.append(transform)
        if pyramid_downsample:
            scale /= pyramid_downsample

    if ome_version == '0.4':
        format = ome_zarr.format.FormatV04()
    elif ome_version == '0.5':
        format = ome_zarr.format.FormatV05()
    else:
        format = ome_zarr.format.CurrentFormat()

    zarr_root = zarr.open_group(store=filename, mode="w", zarr_version=zarr_version)
    write_image(image=data, group=zarr_root, axes=axes, coordinate_transformations=coordinate_transformations,
                scaler=scaler, storage_options=storage_options, fmt=format)

    keys = list(zarr_root.array_keys())
    data_smallest = zarr_root.get(keys[-1])

    # get smallest size image
    zarr_root.attrs['omero'] = create_channel_ome_metadata(data_smallest, dimension_order, channels, ome_version)
