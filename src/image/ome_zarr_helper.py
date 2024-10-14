import zarr
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image

from src.image.util import get_max_downsamples, create_compression_filter
from src.ome_zarr_util import create_axes_metadata, create_transformation_metadata, create_channel_ome_metadata


def save_ome_zarr(filename, data, dimension_order, pixel_size, channels, translation,
                  tile_size=(256, 256), compression=None,
                  npyramid_add=0, pyramid_downsample=2, zarr_version=2, ome_version='0.4'):
    npyramid_add = get_max_downsamples(data.shape, npyramid_add, pyramid_downsample)

    storage_options = {'dimension_separator': '/', 'chunks': tile_size}

    compressor, compression_filters = create_compression_filter(compression)
    if compressor is not None:
        storage_options['compressor'] = compressor
    if compression_filters is not None:
        storage_options['filters'] = compression_filters

    axes = create_axes_metadata(dimension_order)

    coordinate_transformations = []
    scale = 1
    for i in range(npyramid_add + 1):
        coordinate_transformations.append(create_transformation_metadata(dimension_order, pixel_size, scale, translation))
        if pyramid_downsample:
            scale /= pyramid_downsample

    zarr_root = zarr.open_group(store=filename, mode="w", zarr_version=zarr_version)

    write_image(image=data, group=zarr_root, axes=axes, coordinate_transformations=coordinate_transformations,
                scaler=Scaler(downscale=pyramid_downsample, max_layer=npyramid_add),
                storage_options=storage_options)

    keys = list(zarr_root.array_keys())
    data_smallest = zarr_root.get(keys[-1])

    # get smallest size image
    zarr_root.attrs['omero'] = create_channel_ome_metadata(data_smallest, dimension_order, channels, ome_version)
