from dask.diagnostics import ProgressBar
import json
import logging
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import multiview_stitcher
from multiscale_spatial_image import MultiscaleSpatialImage
from multiview_stitcher import registration, fusion, msi_utils, vis_utils, param_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os
import re
from scipy.spatial.transform import Rotation
from spatial_image import SpatialImage
from tqdm import tqdm

from src.OmeZarr import OmeZarr
from src.OmeZarrSource import OmeZarrSource
from src.TiffSource import TiffSource
from src.image.util import *
from src.util import *


mpl.rcParams['figure.dpi'] = 300


def create_example_tiles():
    tile_translations = [
        {"z": 2.5, "y": -10, "x": 30},
        {"z": 2.5, "y": 30, "x": 10},
        {"z": 2.5, "y": 30, "x": 50},
    ]
    dim_order = "czyx"
    scale = {"z": 2, "y": 0.5, "x": 0.5}
    channels = ["DAPI", "GFP"]

    tiles = [{'dim_order': dim_order,
              'translation': tile_translation,
              'scale': scale,
              'channels': channels,
              'data': np.random.randint(0, 100, (2, 10, 100, 100))}
             for tile_translation in tile_translations]
    return tiles


def create_source(filename):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        source = TiffSource(filename)
    elif ext.startswith('zar'):
        source = OmeZarrSource(filename)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    return source


def init_tiles(files, flatfield_quantile=None, invert_x_coordinates=False):
    tiles = []
    sources = [create_source(file) for file in files]
    nchannels = sources[0].get_nchannels()
    images = []
    for source in tqdm(sources):
        output_order = 'yx'
        if source.get_nchannels() > 1:
            output_order += 'c'
        image = redimension_data(source.get_source_dask()[0],
                                 source.dimension_order, output_order)
        images.append(image)

    if flatfield_quantile is not None:
        print('Applying flatfield correction...')
        norm_images = create_normalisation_images(images, quantiles=[flatfield_quantile], nchannels=nchannels)
        dtype = images[0].dtype
        max_image = norm_images[0]
        maxval = 2 ** (8 * dtype.itemsize) - 1
        max_image = max_image / np.float32(maxval)
        images = [float2int_image(flatfield_correction(int2float_image(image), bright=max_image), dtype) for image in images]

    for source, image in zip(sources, images):
        translation = convert_xyz_to_dict(get_value_units_micrometer(source.position))
        if invert_x_coordinates:
            translation['x'] = -translation['x']
        translation['y'] = -translation['y']
        # transform #dimensions need to match
        scale = convert_xyz_to_dict(source.get_pixel_size_micrometer())
        if not translation.keys() == scale.keys():
            translation = {key: translation.get(key, 0) for key in scale.keys()}
        tile = {'dim_order': output_order,
                'translation': translation,
                'scale': scale,
                'channels': source.get_channels(),
                'data': image}
        tiles.append(tile)
    return tiles


def fix_missing_rotation(tiles, source0):
    positions = [np.array(list(tile['translation'].values())) for tile in tiles]
    positions_centre = np.mean(positions, 0)
    center_index = np.argmin([math.dist(position, positions_centre) for position in positions])
    center_position = positions[center_index]
    pairs = get_orthogonal_pairs_from_msims(tiles, source0)
    angles = []
    rotations = []
    for pair in pairs:
        vector = positions[pair[1]] - positions[pair[0]]
        angle = np.rad2deg(np.arctan(vector[1] / vector[0]))
        angles.append(angle)
        rotation = angle
        if rotation < -45:
            rotation += 90
        if rotation > 45:
            rotation -= 90
        rotations.append(rotation)
    rotation = np.mean(rotations)
    transform = create_transform(center=center_position, angle=rotation)
    for tile in tiles:
        keys = list(tile['translation'].keys())
        position = np.array(list(tile['translation'].values()))
        position = apply_transform([position], transform)[0]
        tile['translation'] = {dim: value for dim, value in zip(keys, position)}


def images_to_msims(tiles):
    # build input for stitching
    msims = []
    for tile in tqdm(tiles):
        # input data (can be any numpy compatible array: numpy, dask, cupy, etc.)
        channel_labels = [channel.get('label', '') for channel in tile['channels']]
        sim = si_utils.get_sim_from_array(
            tile['data'],
            dims=list(tile['dim_order']),
            scale=tile['scale'],
            translation=tile['translation'],
            transform_key="stage_metadata",
            c_coords=channel_labels
        )
        msims.append(msi_utils.get_msim_from_sim(sim))
    return msims


def get_orthogonal_pairs_from_msims(images, source0=None):
    """
    Get pairs of orthogonal neighbors from a list of msims.
    This assumes that the msims are placed on a regular grid.
    """

    # get positions (image origins) of msims to be registered
    if isinstance(images[0], MultiscaleSpatialImage):
        sim0 = msi_utils.get_sim_from_msim(images[0])
        spatial_dims = si_utils.get_spatial_dims_from_sim(sim0)
        size = [sim0.sizes[dim] * si_utils.get_spacing_from_sim(sim0)[dim] for dim in spatial_dims]
        origins = np.array([[si_utils.get_origin_from_sim(msi_utils.get_sim_from_msim(msim))[dim] for dim in spatial_dims]
                            for msim in images])
    else:
        size = get_value_units_micrometer(source0.get_physical_size())
        origins = np.array([(tile['translation']['x'], tile['translation']['y']) for tile in images])
    #threshold = [np.mean(np.diff(np.sort(origins[:, dimi]))) for dimi in range(len(spatial_dims))]
    threshold = np.array(size)
    threshold_half = threshold / 2

    # get pairs of neighboring msims
    pairs = []
    for i, j in np.transpose(np.triu_indices(len(images), 1)):
        relvec = abs(origins[i] - origins[j])
        if np.any(relvec < threshold_half) and np.all(relvec < threshold):
            pairs.append((i, j))

    return pairs


def register(sims, msims, reg_channel=None, reg_channel_index=None, filter_foreground=False,
             use_orthogonal_pairs=False, use_rotation=False, fuse_all=True):
    if isinstance(reg_channel, int):
        reg_channel_index = reg_channel
        reg_channel = None

    spatial_dims = si_utils.get_spatial_dims_from_sim(sims[0])

    if filter_foreground:
        print('Filtering foreground tiles...')
        tile_vars = [np.asarray(np.std(sim)).item() for sim in sims]
        threshold = np.median(tile_vars)
        foregrounds = (tile_vars > threshold)
        foreground_msims = [msim for msim, foreground in zip(msims, foregrounds) if foreground]
        not_foreground_msims = [msim for msim, foreground in zip(msims, foregrounds) if not foreground]
        print(f'Foreground tiles: {len(foreground_msims)} / {len(msims)}')

        # duplicate transform keys
        #for msim in not_foreground_msims:
        #    transform = msi_utils.get_transform_from_msim(msim, 'stage_metadata')
        #    msi_utils.set_affine_transform(msim, transform, 'translation_registered')
        for msim in not_foreground_msims:
            msi_utils.set_affine_transform(
                msim,
                param_utils.identity_transform(ndim=2, t_coords=[0]),
                transform_key='registered',
                base_transform_key='stage_metadata')

        indices = np.where(foregrounds)[0]
        register_msims = foreground_msims
    else:
        indices = range(len(msims))
        register_msims = msims

    print('Registering...')
    progress = tqdm()

    if use_orthogonal_pairs:
        pairs = get_orthogonal_pairs_from_msims(register_msims)
    else:
        pairs = None
    with ProgressBar():
        if use_rotation:
            # phase shift registration
            mappings1 = registration.register(
                register_msims,
                reg_channel=reg_channel,
                reg_channel_index=reg_channel_index,
                transform_key='stage_metadata',
                new_transform_key='translation_registered',
                pairs=pairs,
                pre_registration_pruning_method=None,
                groupwise_resolution_kwargs={
                    'transform': 'translation',
                },
                plot_summary=True
            )
            # affine registration
            mappings2 = registration.register(
                register_msims,
                reg_channel=reg_channel,
                reg_channel_index=reg_channel_index,
                transform_key='translation_registered',
                new_transform_key='registered',
                pairs=pairs,
                pre_registration_pruning_method=None,
                pairwise_reg_func=registration.registration_ANTsPy,
                pairwise_reg_func_kwargs={
                    'transform_types': ['Rigid'],  # could also add 'Affine'
                },
                groupwise_resolution_kwargs={
                    'transform': 'rigid',  # could also be affine
                },
                plot_summary=True
            )
            mappings = mappings2
        else:
            mappings = registration.register(
               register_msims,
               reg_channel=reg_channel,
               reg_channel_index=reg_channel_index,
               transform_key="stage_metadata",
               new_transform_key="registered",
               pairs=pairs,
               pre_registration_pruning_method=None,
               plot_summary=True
            )

    progress.update()
    progress.close()
    mappings_dict = {int(index): mapping.data.tolist() for index, mapping in zip(indices, mappings)}

    print('Fusing...')
    if fuse_all:
        fuse_msims = msims
    else:
        fuse_msims = register_msims
    fused_image = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in fuse_msims],
        transform_key="registered"
    )
    return mappings_dict, fused_image


def save_zarr(filename, image, source):
    #image.to_zarr(filename)    # gives attribute error (creates empty attr dictionary) in xarray.to_zarr
    #msi_utils.multiscale_spatial_image_to_zarr(msi_utils.get_msim_from_sim(image), filename)   # creates 'scaleX' keys
    ##ngff_utils.sim_to_ngff_image(image, filename, source)
    if isinstance(image, SpatialImage):
        source.output_dimension_order = ''.join(image.dims)
    zarr = OmeZarr(filename)
    zarr.write(image.data, source)


def convert_xyz_to_dict(xyz):
    dct = {dim: value for dim, value in zip('xyz', xyz)}
    return dct


def dir_regex(pattern):
    dir = os.path.dirname(pattern)
    file_pattern = os.path.basename(pattern)
    files = [os.path.join(dir, file) for file in os.listdir(dir) if re.search(file_pattern, file)]
    files_sorted = sorted(files, key=lambda file: find_all_numbers(get_filetitle(file)))
    return files_sorted


def show_image(image, title='', cmap=None):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if cmap is None:
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.tight_layout()
    plt.show()


def run_simple():
    print(f'Multiview-stitcher Version: {multiview_stitcher.__version__}')

    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/subselection/tiles_1_MMStack_New Grid 1-Grid_(?!0_0.ome.tif).*'     # 3x3 subselection
    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/tiles_1_MMStack_New Grid 1-Grid_5_.*.ome.tif'     # one column of tiles
    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    #input = 'D:/slides/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/EM04768_01_sub_04_fluorescence_10x/converted/.*.ome.tif'
    #input = '/nemo/project/proj-czi-vp/raw/lm/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'

    #input = ['output_reflect_orth/registered.ome.zarr', 'output_fluor_orth/registered.ome.zarr']

    #input = 'D:/slides/EM04768_01_substrate_04/EM/a0004/roi0000/t.*/.*.ome.tif'
    input = 'D:/slides/EM04768_01_substrate_04/EM/a0004/roi0000/.*.ome.tif'

    #invert_x_coordinates = True
    #flatfield_quantile = 0.95
    #filter_foreground = True
    #use_orthogonal_pairs = True

    invert_x_coordinates = False
    flatfield_quantile = None
    filter_foreground = False
    use_orthogonal_pairs = False
    use_rotation = False

    reg_channel = 0

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_tiles_filename = os.path.join(output_dir, 'tiles_original.png')
    original_fused_filename = os.path.join(output_dir, 'original.ome.zarr')
    registered_tiles_filename = os.path.join(output_dir, 'tiles_registered.png')
    registered_fused_filename = os.path.join(output_dir, 'registered.ome.zarr')

    mvsr_logger = logging.getLogger('multiview_stitcher.registration')
    mvsr_logger.setLevel(logging.INFO)
    if len(mvsr_logger.handlers) == 0:
        mvsr_logger.addHandler(logging.StreamHandler())

    print('Initialising tiles...')
    if isinstance(input, list):
        filenames = input
        file_indices = list(range(len(filenames)))
    else:
        filenames = dir_regex(input)
        file_indices = ['-'.join(map(str, find_all_numbers(get_filetitle(filename))[-2:])) for filename in filenames]
    source0 = create_source(filenames[0])
    tiles = init_tiles(filenames, flatfield_quantile=flatfield_quantile, invert_x_coordinates=invert_x_coordinates)
    # hack
    fix_missing_rotation(tiles, source0)

    print('Converting tiles...')
    msims = images_to_msims(tiles)
    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    # before registration:
    print('Fusing original...')
    original_fused = fusion.fuse(
        sims,
        transform_key='stage_metadata'
    )

    # plot the tile configuration
    print('Plotting tiles...')
    vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=False, output_filename=original_tiles_filename)

    print('Saving fused image...')
    save_zarr(original_fused_filename, original_fused, source0)
    #show_image(original_fused.data[0, 0, ...])

    mappings, registered_fused = register(sims, msims, reg_channel,
                                          filter_foreground=filter_foreground,
                                          use_orthogonal_pairs=use_orthogonal_pairs,
                                          use_rotation=use_rotation)
    mappings2 = {get_filetitle(filenames[index]): mapping for index, mapping in mappings.items()}
    with open(os.path.join(output_dir, 'mappings.json'), 'w') as file:
        json.dump(mappings2, file, indent=4)

    # plot the tile configuration after registration
    print('Plotting tiles...')
    vis_utils.plot_positions(msims, transform_key='registered', use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=False, output_filename=registered_tiles_filename)

    print('Saving fused image...')
    save_zarr(registered_fused_filename, registered_fused, source0)
    #show_image(registered_fused.data[0, 0, 5, ...]) # XYZ example data - show middle of Z depth
    #show_image(registered_fused.data[0, 0, ...])


def run():
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/subselection/tiles_1_MMStack_New Grid 1-Grid_(?!0_0.ome.tif).*'     # 3x3 subselection
    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    input = '/nemo/project/proj-czi-vp/raw/lm/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    filenames = dir_regex(input)
    source0 = create_source(filenames[0])
    tiles = init_tiles(filenames, flatfield_quantile=0.95, invert_x_coordinates=True)
    msims = images_to_msims(tiles)
    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]
    mappings1, registered_fused1 = register(sims, msims, 0,
                                            filter_foreground=True,
                                            use_orthogonal_pairs=True,
                                            use_rotation=False,
                                            fuse_all=False)

    #input = 'D:/slides/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/subselection/tiles_1_MMStack_New Grid 1-Grid_(?!0_0.ome.tif).*'  # 3x3 subselection
    #input = 'D:/slides/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/EM04768_01_sub_04_fluorescence_10x/converted/.*.ome.tif'
    input = '/nemo/project/proj-czi-vp/raw/lm/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/EM04768_01_sub_04_fluorescence_10x/converted/.*.ome.tif'
    filenames = dir_regex(input)
    tiles = init_tiles(filenames, flatfield_quantile=0.95, invert_x_coordinates=True)
    msims = images_to_msims(tiles)
    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]
    mappings2, registered_fused2 = register(sims, msims, 0,
                                            filter_foreground=True,
                                            use_orthogonal_pairs=True,
                                            use_rotation=False,
                                            fuse_all=False)

    sims = [registered_fused1, registered_fused2]
    msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]
    # set dummy position
    for msim in msims:
        msi_utils.set_affine_transform(msim,
                                       param_utils.identity_transform(ndim=2, t_coords=[0]),
                                       transform_key='stage_metadata')
    mappings, registered_fused = register(sims, msims, 0,
                                          filter_foreground=False,
                                          use_orthogonal_pairs=False,
                                          use_rotation=False)
    with open(os.path.join(output_dir, 'mappings_overlay.json'), 'w') as file:
        json.dump(mappings, file, indent=4)
    registered_tiles_filename = os.path.join(output_dir, 'overlay_registered.png')
    registered_fused_filename = os.path.join(output_dir, 'registered.ome.zarr')

    # plot the tile configuration after registration
    print('Plotting tiles...')
    vis_utils.plot_positions(msims, transform_key='registered', use_positional_colors=False,
                             show_plot=False, output_filename=registered_tiles_filename)

    print('Saving fused image...')
    save_zarr(registered_fused_filename, registered_fused, source0)
    # TODO: save channels to zarr using mappings offset


if __name__ == '__main__':
    run_simple()
    print('Done!')
