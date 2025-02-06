from datetime import datetime
import dask.array as da
from multiview_stitcher import spatial_image_utils as si_utils, fusion
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import xarray as xr

from src.image.ome_helper import save_image
from src.image.ome_tiff_helper import save_ome_tiff
from src.TiffSource import TiffSource
from src.registration import init_tiles
from src.util import *
from src.image.util import *


def test_channels(filename):
    channels = [{'label': 'Reflection', 'color': (1, 1, 1)},
                {'label': 'Fluorescence', 'color': (0, 1, 0)}]

    data = xr.DataArray(
        da.random.randint(0, 65535, size=(2, 1024, 1024), chunks=(1, 256, 256), dtype=da.uint16),
        dims=list('cyx'),
        coords={'c': [channel['label'] for channel in channels]},
        attrs={'channels': channels},
    )
    #data.assign_coords({'c': ['test_channel']})
    #data.assign_attrs({'channels': channels})

    sim = si_utils.get_sim_from_array(data, dims=data.dims,
                                      scale=convert_xyz_to_dict([0.1, 0.1]),
                                      translation=convert_xyz_to_dict([1.2, 3.4]))

    save_image(filename, sim, channels=channels)


def test_zstack(filename):
    data = xr.DataArray(
        da.random.randint(0, 65535, size=(10, 1024, 1024), chunks=(1, 256, 256), dtype=da.uint16),
        dims=list('zyx')
    )

    sim = si_utils.get_sim_from_array(data, dims=data.dims,
                                      scale=convert_xyz_to_dict([0.1, 0.1, 0.5]),
                                      translation=convert_xyz_to_dict([1.2, 3.4, 5.6]))

    save_image(filename, sim)


def test_pipeline(tmp_path, nfiles=2):
    dimension_order = 'yx'
    size = (1024, 1024)
    chunks = (256, 256)
    pixel_size = [0.1, 0.1]
    z_scale = 0.5

    filenames = []
    z_position = 0
    for filei in range(nfiles):
        filename = tmp_path / datetime.now().strftime('%Y%m%d_%H%M%S_%f.ome.tiff')
        position = list(np.random.rand(2)) + [z_position]
        z_position += z_scale
        data = xr.DataArray(
            da.random.randint(0, 65535, size=size, chunks=chunks, dtype=da.uint16),
            dims=list(dimension_order)
        )
        save_ome_tiff(filename, data.data, dimension_order, pixel_size, positions=[position])
        filenames.append(filename)

    sims, translations, rotations = init_tiles(filenames)

    output_stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
    stack_sims = [fusion.fuse(
        [sim],
        transform_key='stage_metadata',
        output_stack_properties=output_stack_properties
    ) for sim in sims]
    #fused_image = xr.combine_nested([sim.rename() for sim in sims], concat_dim='z', combine_attrs='override')
    fused_image = xr.combine_nested([sim.rename() for sim in stack_sims], concat_dim='z', combine_attrs='override')

    save_image(tmp_path / 'fused', fused_image, transform_key='stage_metadata', params={'format': 'tif'})
    print(tmp_path)
    pass



if __name__ == '__main__':
    #filename = 'output/test'
    #test_channels(filename)
    #test_zstack(filename)
    #source = TiffSource(filename + '.ome.tiff')
    #print(source.get_pixel_size())
    #print(source.get_position())
    #print(source.get_channels())

    with TemporaryDirectory() as temp_dir:
        test_pipeline(Path(temp_dir))
