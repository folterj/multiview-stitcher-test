from multiview_stitcher import registration, vis_utils
from pathlib import Path
from tempfile import TemporaryDirectory
import xarray as xr
from tqdm import tqdm

from src.image.ome_helper import save_image
from src.image.source_helper import create_source
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


def calc_shape(y, x, offset):
    h, w = y.shape
    xoffset, yoffset = offset
    image = (1 + np.cos((xoffset + x / w) * 2 * np.pi)
             * np.cos((yoffset + y / h) * 2 * np.pi)) / 2
    image[image < 0.01] = 1
    return image


def init_tiles_pattern(n=2, size=(1024, 1024), chunks=(256, 256), dimension_order0='yx', transform_key='stage_metadata'):
    z_scale = 0.5
    pixel_size = [0.1, 0.1, z_scale]
    dtype = np.dtype(np.uint16)

    dimension_order = dimension_order0
    if 'z' not in dimension_order:
        # add z axis to store z position
        dimension_order = 'z' + dimension_order

    if not isinstance(chunks, dict):
        chunks = convert_xyz_to_dict(chunks, dimension_order)

    z_position = 0
    offset = [0, 0]
    sims = []
    for _ in tqdm(range(n), desc='Init tiles'):
        position = list(np.random.rand(2)) + [z_position]
        shape_image = np.fromfunction(calc_shape, tuple(size), dtype=np.float32, offset=offset)
        noise_image = np.random.random_sample(size)
        pattern_image = float2int_image(
            np.clip(0.9 * shape_image + 0.1 * noise_image, 0, 1),
            target_dtype=dtype)
        pattern_image = redimension_data(pattern_image, dimension_order0, dimension_order)
        sim = si_utils.get_sim_from_array(
            pattern_image,
            dims=list(dimension_order),
            scale=convert_xyz_to_dict(pixel_size),
            translation=convert_xyz_to_dict(position),
            transform_key=transform_key
        )
        sims.append(sim.chunk(chunks))
        z_position += z_scale
        offset[0] += 0.05
        offset[1] -= 0.01
    return sims


def init_sim(data, chunks=(1024, 1024), dimension_order='yx', position=None, pixel_size=None, transform_key='stage_metadata'):
    if not isinstance(chunks, dict):
        chunks = convert_xyz_to_dict(chunks)
    sim = si_utils.get_sim_from_array(
        data,
        dims=list(dimension_order),
        scale=convert_xyz_to_dict(pixel_size),
        translation=convert_xyz_to_dict(position),
        transform_key=transform_key
    )
    return sim.chunk(chunks)


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


def test5(sims, tmp_path):
    transform_key = 'stage_metadata'
    new_transform_key = 'registered'

    #from src.registration_methods.RegistrationMethodCPD import RegistrationMethodCPD
    #registration_method = RegistrationMethodCPD(sims[0].dtype)
    #pairwise_reg_func = registration_method.registration
    pairwise_reg_func = registration.phase_correlation_registration

    # register in 2D
    # pairs: pairwise consecutive views
    reg_sims = [si_utils.max_project_sim(sim, dim='z') for sim in sims]
    reg_msims = [msi_utils.get_msim_from_sim(sim) for sim in reg_sims]
    progress = tqdm(desc='Register', total=1)
    params = registration.register(
        reg_msims,
        reg_channel=sims[0].coords['c'].values[0],
        transform_key=transform_key,
        new_transform_key=new_transform_key,
        pairs = [(i, i+1) for i in range(len(sims)-1)],
        pairwise_reg_func=pairwise_reg_func
    )
    progress.update()
    progress.close()
    for param in params:
        print(param.data[0].tolist())

    # set 3D affine transforms from 2D registration params
    for index, sim in enumerate(sims):
        affine_3d = param_utils.identity_transform(ndim=3)
        affine_3d.loc[{dim: params[index].coords[dim] for dim in params[index].sel(t=0).dims}] = params[index].sel(t=0)
        si_utils.set_sim_affine(sim, affine_3d, transform_key=new_transform_key, base_transform_key=transform_key)

    # continue with new transform key
    transform_key = new_transform_key

    print(f'New transforms shape: {sims[0].transforms[new_transform_key].shape}')
    progress = tqdm(desc='Plot', total=1)
    vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim) for sim in sims], transform_key=transform_key, use_positional_colors=False)
    progress.update()
    progress.close()

    z_scale = 0.5

    # set output spacing
    output_spacing = si_utils.get_spacing_from_sim(sims[0]) | {'z': z_scale}

    # calculate output stack properties from input views
    output_stack_properties = fusion.calc_stack_properties_from_view_properties_and_params(
        [si_utils.get_stack_properties_from_sim(sim) for sim in sims],
        [np.array(si_utils.get_affine_from_sim(sim, transform_key).squeeze()) for sim in sims],
        output_spacing,
        mode='union',
    )

    # convert to dict form
    output_stack_properties = {
        k: {dim: v[idim] for idim, dim in enumerate(output_spacing.keys())}
        for k, v in output_stack_properties.items()
    }

    # set z shape which is wrongly calculated by calc_stack_properties_from_view_properties_and_params
    # because it does not take into account the correct input z spacing because of stacks of one z plane
    output_stack_properties['shape']['z'] = len(sims)

    data_size = np.prod(list(output_stack_properties['shape'].values())) * sims[0].dtype.itemsize
    print(f'Fused size {print_hbytes(data_size)}')

    progress = tqdm(desc='Fuse', total=1)
    # fuse all sims together using simple average fusion
    fused_image = fusion.fuse(
        sims,
        transform_key=transform_key,
        output_stack_properties=output_stack_properties,
        output_chunksize={'z': 1, 'y': 1024, 'x': 1024},
        fusion_func=fusion.simple_average_fusion,
    )
    progress.update()
    progress.close()

    progress = tqdm(desc='Save zarr', total=1)
    save_image(tmp_path / 'fused', fused_image, transform_key=transform_key, params={'format': 'zar'})
    progress.update()
    progress.close()

    progress = tqdm(desc='Save tiff', total=1)
    save_image(tmp_path / 'fused', fused_image, transform_key=transform_key, params={'format': 'tif'})
    progress.update()
    progress.close()


def test_pipeline(tmp_path, n=2):
    size = (1000, 1000)
    chunks = (1024, 1024)
    print('size:', size)
    print('chunks:', chunks)
    sims = init_tiles_pattern(n, size=size, chunks=chunks)
    test5(sims, tmp_path)


def create_stack(path, n=100):
    dtype = np.dtype(np.uint16)
    size = (10000, 10000)
    chunks = (1024, 1024)
    z_scale = 0.5
    pixel_size = [0.1, 0.1, z_scale]
    dimension_order0 = 'yx'
    dimension_order = 'zyx'
    transform_key = 'stage_metadata'

    noise_image = float2int_image(np.random.random_sample(size), target_dtype=dtype)
    data = redimension_data(noise_image, dimension_order0, dimension_order)

    z = 0
    for index in tqdm(range(n), desc='Creating tiles'):
        filename = str(path / f'image{index:04}')
        sim = init_sim(data, chunks=chunks, dimension_order='zyx', pixel_size=pixel_size, position=[0, 0, z], transform_key = 'stage_metadata')
        save_image(filename, sim, transform_key=transform_key, params={'format': 'zar'})
        z += z_scale


def test_create_stack(path, n):
    transform_key = 'stage_metadata'
    z_scale = 0.5

    sims = []
    for index in tqdm(range(n), desc='Init tiles'):
        filename = str(path / f'image{index:04}')
        source = create_source(filename + '.ome.zarr')
        dimension_order = source.dimension_order
        sim = si_utils.get_sim_from_array(
            source.get_source_dask()[0],
            dims=list(dimension_order),
            scale=convert_xyz_to_dict(source.get_pixel_size_micrometer()),
            translation=convert_xyz_to_dict(source.get_position_micrometer()),
            transform_key=transform_key
        )
        sims.append(sim)

    # set output spacing
    output_spacing = si_utils.get_spacing_from_sim(sims[0]) | {'z': z_scale}

    # calculate output stack properties from input views
    output_stack_properties = fusion.calc_stack_properties_from_view_properties_and_params(
        [si_utils.get_stack_properties_from_sim(sim) for sim in sims],
        [np.array(si_utils.get_affine_from_sim(sim, transform_key).squeeze()) for sim in sims],
        output_spacing,
        mode='union',
    )

    # convert to dict form
    output_stack_properties = {
        k: {dim: v[idim] for idim, dim in enumerate(output_spacing.keys())}
        for k, v in output_stack_properties.items()
    }

    # set z shape which is wrongly calculated by calc_stack_properties_from_view_properties_and_params
    # because it does not take into account the correct input z spacing because of stacks of one z plane
    output_stack_properties['shape']['z'] = len(sims)

    data_size = np.prod(list(output_stack_properties['shape'].values())) * sims[0].dtype.itemsize
    print(f'Fusing {print_hbytes(data_size)}')

    progress = tqdm(desc='Fuse', total=1)
    # fuse all sims together using simple average fusion
    fused_image = fusion.fuse(
        sims,
        transform_key=transform_key,
        output_stack_properties=output_stack_properties,
        output_chunksize={'z': 1, 'y': 1024, 'x': 1024},
        fusion_func=fusion.simple_average_fusion,
    )
    progress.update()
    progress.close()

    progress = tqdm(desc='Save zarr', total=1)
    save_image(path / 'fused', fused_image, transform_key=transform_key, params={'format': 'zar'})
    progress.update()
    progress.close()



if __name__ == '__main__':
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir)

        path = Path('D:/slides/test_stack/')
        #create_stack(path, n=100)
        #test_create_stack(path, n=100)
        test_pipeline(path)
