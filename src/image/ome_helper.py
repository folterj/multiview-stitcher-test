from ome_zarr.scale import Scaler

from src.image.ome_ngff_helper import save_ome_ngff
from src.image.ome_tiff_helper import save_ome_tiff
from src.image.ome_zarr_helper import save_ome_zarr
from src.image.util import *


def save_image(filename, sim, transform_key=None, channels=None, translation0=None, params={}):
    dimension_order = ''.join(sim.dims)
    sdims = ''.join(si_utils.get_spatial_dims_from_sim(sim))
    sdims = sdims.replace('zyx', 'xyz').replace('yx', 'xy')   # order xy(z)
    pixel_size = []
    for dim in sdims:
        pixel_size1 = si_utils.get_spacing_from_sim(sim)[dim]
        if pixel_size1 == 0:
            pixel_size1 = 1
        pixel_size.append(pixel_size1)
    # metadata: only use coords of fused image
    position, rotation = get_data_mapping(sim, transform_key=transform_key, translation0=translation0)
    nplanes = sim.sizes.get('z', 1) * sim.sizes.get('c', 1) * sim.sizes.get('t', 1)
    positions = [position] * nplanes

    if channels is None:
        channels = sim.attrs.get('channels', [])

    tile_size = params.get('tile_size')
    if tile_size is not None:
        chunking = retuple(tuple(reversed(tile_size)), sim.shape)
        sim = sim.chunk(chunks=chunking)

    compression = params.get('compression')
    pyramid_downsample = params.get('pyramid_downsample', 2)
    npyramid_add = get_max_downsamples(sim.shape, params.get('npyramid_add', 0), pyramid_downsample)
    scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)

    if 'zar' in params.get('format', 'zar'):
        #save_ome_zarr(f'{filename}.ome.zarr', sim.data, dimension_order, pixel_size,
        #              channels, position, rotation, compression=compression, scaler=scaler)
        save_ome_ngff(f'{filename}.ome.zarr', sim, channels, position, rotation,
                      pyramid_downsample=pyramid_downsample)
    if 'tif' in params.get('format', 'tif'):
        save_ome_tiff(f'{filename}.ome.tiff', sim.data, dimension_order, pixel_size,
                      channels, positions, rotation, tile_size=tile_size, compression=compression, scaler=scaler)
