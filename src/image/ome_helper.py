from ome_zarr.scale import Scaler

from src.image.ome_tiff_helper import save_ome_tiff
from src.image.ome_zarr_helper import save_ome_zarr
from src.image.util import *


def save_image(filename, sim, transform_key=None, channels=None, translation0=None,
               npyramid_add=4, pyramid_downsample=2, params={}):
    dimension_order = ''.join(sim.dims)
    sdims = ''.join(si_utils.get_spatial_dims_from_sim(sim))
    sdims = sdims.replace('zyx', 'xyz').replace('yx', 'xy')   # order xy(z)
    pixel_size = [si_utils.get_spacing_from_sim(sim)[dim] for dim in sdims]
    # metadata: only use coords of fused image
    position, rotation = get_data_mapping(sim, transform_key=transform_key,
                                          translation0=translation0)
    nplanes = sim.sizes.get('z', 1) * sim.sizes.get('c', 1) * sim.sizes.get('t', 1)
    positions = [position] * nplanes

    if channels is None:
        channels = sim.attrs.get('channels', [])

    npyramid_add = get_max_downsamples(sim.shape, npyramid_add, pyramid_downsample)
    scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)

    if 'zar' in params.get('format', 'zar'):
        save_ome_zarr(f'{filename}.ome.zarr', sim.data, dimension_order, pixel_size, channels, position, rotation, scaler=scaler)
    if 'tif' in params.get('format', 'tif'):
        save_ome_tiff(f'{filename}.ome.tiff', sim.data, dimension_order, pixel_size, channels, positions, rotation, scaler=scaler)
