import dask.array as da
from multiview_stitcher import fusion
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
import xarray as xr

from src.image.util import *


def convert_xyz_to_dict(xyz, axes='xyz'):
    dct = {dim: value for dim, value in zip(axes, xyz)}
    return dct


def redimension_data(data, old_order, new_order, **indices):
    # able to provide optional dimension values e.g. t=0, z=0
    if new_order == old_order:
        return data

    new_data = data
    order = old_order
    # remove
    for o in old_order:
        if o not in new_order:
            index = order.index(o)
            dim_value = indices.get(o, 0)
            new_data = np.take(new_data, indices=dim_value, axis=index)
            order = order[:index] + order[index + 1:]
    # add
    for o in new_order:
        if o not in order:
            new_data = np.expand_dims(new_data, 0)
            order = o + order
    # move
    old_indices = [order.index(o) for o in new_order]
    new_indices = list(range(len(new_order)))
    new_data = np.moveaxis(new_data, old_indices, new_indices)
    return new_data


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


def test(sims, tmp_path):
    z_scale = 0.5
    output_stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
    if z_scale is not None:
        output_stack_properties['spacing']['z'] = z_scale
    stack_sims = [fusion.fuse(
        [sim],
        transform_key='stage_metadata',
        output_stack_properties=output_stack_properties,
        fusion_func=fusion.simple_average_fusion,
        #output_chunksize={'z': 1, 'y': 1000, 'x': 1000},
    ) for sim in sims]
    fused_image = xr.combine_nested([sim.rename() for sim in stack_sims], concat_dim='z', combine_attrs='override')
    spacing = si_utils.get_spacing_from_sim(fused_image)
    print(spacing)
    fused_image.compute()


def test2(sims, tmp_path):
    # error in fuse: fix_dims=[] (instead of ['z']), not fusing plane-wise; division by 0 in edt_support_spacing = {...}
    # z_scale = 0.5
    output_stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
    # if z_scale is not None:
    #     output_stack_properties['spacing']['z'] = z_scale
    stack_sims = [fusion.fuse(
        [sim],
        transform_key='stage_metadata',
        output_stack_properties=output_stack_properties,
        # fusion_func=fusion.simple_average_fusion,
    ) for sim in sims]
    # stack_sims = sims
    fused_image = xr.combine_nested([sim.rename() for sim in stack_sims], concat_dim='z', combine_attrs='override')
    # fused_image.data.compute(scheduler='single-threaded')
    import matplotlib.pyplot as plt
    import tifffile
    #tifffile.imshow(fused_image.data)
    #plt.show()
    show_image(fused_image[0][0][0])
    show_image(fused_image[0][0][1])


def test3(sims, tmp_path):
    # error in fuse: fix_dims=[] (instead of ['z']), not fusing plane-wise; division by 0 in edt_support_spacing = {...}
    # z_scale = 0.5
    output_stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
    # if z_scale is not None:
    #     output_stack_properties['spacing']['z'] = z_scale

    stack_sims = [fusion.fuse(
        [sim],
        transform_key='stage_metadata',
        output_stack_properties=output_stack_properties,
        # fusion_func=fusion.simple_average_fusion,
    ) for sim in sims]

    fused_image = xr.combine_nested(stack_sims, concat_dim='z')
    #fused_image = xr.concat(stack_sims, dim='z')

    show_image(sims[0][0][0][0])
    show_image(sims[1][0][0][0])

    show_image(stack_sims[0][0][0][0])
    show_image(stack_sims[1][0][0][0])


def test4(sims, tmp_path):
    z_scale = 0.5
    transform_key = 'stage_metadata'

    # set output spacing
    output_spacing = si_utils.get_spacing_from_sim(sims[0]) | {'z': z_scale}

    # calculate output stack properties from input views
    output_stack_properties = fusion.calc_stack_properties_from_view_properties_and_params(
        [si_utils.get_stack_properties_from_sim(sim) for sim in sims],
        [np.array(si_utils.get_affine_from_sim(sim, transform_key).squeeze()) for sim in sims],
        output_spacing,
        mode='union',
    )

    # convert to dict form (this should not be needed anymore in the next release)
    output_stack_properties = {
        k: {dim: v[idim] for idim, dim in enumerate(output_spacing.keys())}
        for k, v in output_stack_properties.items()
    }

    # set z shape which is wrongly calculated by calc_stack_properties_from_view_properties_and_params
    # because it does not take into account the correct input z spacing because of stacks of one z plane
    output_stack_properties['shape']['z'] = len(sims)

    # fuse all sims together using simple average fusion
    fused_image = fusion.fuse(
        sims,
        transform_key=transform_key,
        output_stack_properties=output_stack_properties,
        #output_chunksize={'z': 1, 'y': 1024, 'x': 1024},
        fusion_func=fusion.simple_average_fusion,
    )
    return fused_image


def test_pipeline(tmp_path, ntiles=2):
    #sims, translations, rotations = test_init_tiles(tmp_path, ntiles)
    sims = test_init_tiles_simple(ntiles)
    test4(sims, tmp_path)
    print(tmp_path)
    pass



if __name__ == '__main__':
    with TemporaryDirectory() as temp_dir:
        test_pipeline(Path(temp_dir))
