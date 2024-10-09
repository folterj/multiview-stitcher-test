from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image


def save_ome_zarr(filename, data, pixel_size, position, channels,
                  tile_size=(256, 256), compression='LZW',
                  npyramid_add=0, pyramid_downsample=2):
    write_image(image=data, group=zarr_root, axes=axes, coordinate_transformations=pixel_size_scales,
                scaler=Scaler(downscale=pyramid_downsample, max_layer=npyramid_add),
                storage_options=storage_options)
