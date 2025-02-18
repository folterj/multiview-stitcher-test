from datetime import datetime
import dask.array as da
from multiview_stitcher import fusion
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
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


def test_init_tiles(tmp_path, ntiles=2):
    dimension_order = 'yx'
    size = (1024, 1024)
    chunks = (256, 256)
    pixel_size = [0.1, 0.1]
    z_scale = 0.5

    filenames = []
    z_position = 0
    for filei in range(ntiles):
        filename = tmp_path / datetime.now().strftime('%Y%m%d_%H%M%S_%f.ome.tiff')
        position = list(np.random.rand(2)) + [z_position]
        z_position += z_scale
        data = xr.DataArray(
            da.random.randint(0, 65535, size=size, chunks=chunks, dtype=da.uint16),
            dims=list(dimension_order)
        )
        save_ome_tiff(filename, data.data, dimension_order, pixel_size, positions=[position])
        filenames.append(filename)

    return init_tiles(filenames, 'stage_metadata')


def test_init_tiles_simple(ntiles=2):
    dimension_order0 = 'yx'
    dimension_order = 'zyx'
    size = (1024, 1024)
    chunks = (256, 256)
    pixel_size = [0.1, 0.1]
    z_scale = 0.5

    z_position = 0
    sims = []
    for filei in range(ntiles):
        position = list(np.random.rand(2)) + [z_position]
        z_position += z_scale
        data = xr.DataArray(
            da.random.randint(0, 65535, size=size, chunks=chunks, dtype=da.uint16),
            dims=list(dimension_order0)
        )
        data = redimension_data(data, dimension_order0, dimension_order)    # always add z axis to store z position
        scale_dict = convert_xyz_to_dict(pixel_size)
        if len(scale_dict) > 0 and 'z' not in scale_dict:
            scale_dict['z'] = 1
        translation_dict = convert_xyz_to_dict(position)
        if len(translation_dict) > 0 and 'z' not in translation_dict:
            translation_dict['z'] = 0
        sim = si_utils.get_sim_from_array(
            data,
            dims=list(dimension_order),
            scale=scale_dict,
            translation=translation_dict,
            transform_key='stage_metadata'
        )
        sims.append(sim)
    return sims


def test1(sims, tmp_path):
    # works
    output_stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
    stack_sims = [fusion.fuse(
        [sim],
        transform_key='stage_metadata',
        output_stack_properties=output_stack_properties
    ) for sim in sims]
    fused_image = xr.combine_nested([sim.rename() for sim in stack_sims], concat_dim='z', combine_attrs='override')
    save_image(tmp_path / 'fused', fused_image, transform_key='stage_metadata', params={'format': 'tif'})

def test2(sims, tmp_path):
    # error in fuse: fix_dims=[] (instead of ['z']), not fusing plane-wise; division by 0 in edt_support_spacing = {...}
    z_scale = 0.5
    output_stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
    if z_scale is not None:
        output_stack_properties['spacing']['z'] = z_scale
    stack_sims = [fusion.fuse(
        [sim],
        transform_key='stage_metadata',
        output_stack_properties=output_stack_properties
    ) for sim in sims]
    fused_image = xr.combine_nested([sim.rename() for sim in stack_sims], concat_dim='z', combine_attrs='override')
    save_image(tmp_path / 'fused', fused_image, transform_key='stage_metadata', params={'format': 'tif'})

def test3(sims, tmp_path):
    # error in fuse: same as in test2
    stack_sims = [fusion.fuse(
        [sim],
        transform_key='stage_metadata',
    ) for sim in sims]
    fused_image = xr.combine_nested([sim.rename() for sim in stack_sims], concat_dim='z', combine_attrs='override')
    save_image(tmp_path / 'fused', fused_image, transform_key='stage_metadata', params={'format': 'tif'})


def test_pipeline(tmp_path, ntiles=2):
    #sims, translations, rotations = test_init_tiles(tmp_path, ntiles)
    sims = test_init_tiles_simple(ntiles)

    try:
        test1(sims, tmp_path)
        print('test 1 ok')
    except Exception:
        print(traceback.format_exc())
        print('test 1 error')

    try:
        test2(sims, tmp_path)
        print('test 2 ok')
    except Exception:
        print(traceback.format_exc())
        print('test 2 error')

    try:
        test3(sims, tmp_path)
        print('test 3 ok')
    except Exception:
        print(traceback.format_exc())
        print('test 3 error')

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
