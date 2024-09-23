import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
import zarr

from src.image.util import *
from src.ome_zarr_util import *
from src.util import *


class OmeZarr:
    def __init__(self, filename):
        self.filename = filename

    def write(self, data, sources, tile_size=[], compression=[],
              npyramid_add=0, pyramid_downsample=2, translations=[],
              image_operations=[]):

        compressor, compression_filters = create_compression_filter(compression)
        storage_options = {'dimension_separator': '/', 'chunks': tile_size}
        ome_version = '0.4'
        # Zarr V3 testing
        #storage_options = {'chunks': tile_size}
        if compressor is not None:
            storage_options['compressor'] = compressor
        if compression_filters is not None:
            storage_options['filters'] = compression_filters
        self.storage_options = storage_options

        zarr_root = zarr.open_group(store=parse_url(self.filename, mode="w").store, mode="w", storage_options=storage_options)
        # Zarr V3 testing
        #zarr_root = zarr.open_group(store=parse_url(self.filename, mode="w").store, mode="w", zarr_version=3)
        root_path = ''

        multiple_images = isinstance(sources, list)
        multi_metadata = []
        if not multiple_images:
            data = [data]
            sources = [sources]

        for index, (data1, source) in enumerate(zip(data, sources)):
            for image_operation in image_operations:
                data1 = image_operation(data1)
            if multiple_images:
                root_path = str(index)
                root = zarr_root.create_group(root_path)
            else:
                root = zarr_root
            if index < len(translations):
                translation_um = translations[index]
            else:
                translation_um = source.get_position_micrometer()
            self.write_dataset(root, data1, source, npyramid_add, pyramid_downsample, translation_um)
            if multiple_images:
                meta = root.attrs['multiscales'][0].copy()
                for dataset_meta in meta['datasets']:
                    dataset_meta['path'] = f'{root_path}/{dataset_meta["path"]}'
                multi_metadata.append(meta)
        if multiple_images:
            zarr_root.attrs['multiscales'] = multi_metadata
        zarr_root.attrs['omero'] = create_channel_metadata(sources[0], ome_version)

    def write_dataset(self, zarr_root, data, source,
                      npyramid_add=0, pyramid_downsample=2, translation_um=[]):

        pixel_size_um = source.get_pixel_size_micrometer()
        if len(pixel_size_um) == 0:
            npyramid_add = 0

        dimension_order = source.get_dimension_order()
        if 'c' in dimension_order and dimension_order.index('c') == len(dimension_order) - 1:
            # ome-zarr doesn't support channel after space dimensions (yet)
            data = np.moveaxis(data, -1, 0)
            dimension_order = dimension_order[-1] + dimension_order[:-1]

        axes = create_axes_metadata(dimension_order)

        pixel_size_scales = []
        scale = 1
        for i in range(npyramid_add + 1):
            pixel_size_scales.append(create_transformation_metadata(dimension_order, pixel_size_um, scale, translation_um))
            if pyramid_downsample:
                scale /= pyramid_downsample

        #write_image(image=data, group=zarr_root, axes=axes, coordinate_transformations=pixel_size_scales,
        #            scaler=Scaler(downscale=pyramid_downsample, max_layer=npyramid_add), overwrite=True)
        # Zarr V3 testing
        write_image(image=data, group=zarr_root, axes=axes, coordinate_transformations=pixel_size_scales,
                    scaler=Scaler(downscale=pyramid_downsample, max_layer=npyramid_add),
                    storage_options=self.storage_options)
