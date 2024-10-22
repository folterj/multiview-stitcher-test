# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

from dask.diagnostics import ProgressBar
import json
import logging
import math
import multiview_stitcher
from multiview_stitcher import registration, fusion, msi_utils, vis_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os
import re
from ome_zarr.scale import Scaler
from tqdm import tqdm
import xarray as xr

from src.OmeZarrSource import OmeZarrSource
from src.TiffSource import TiffSource
from src.image.ome_tiff_helper import save_ome_tiff
from src.image.ome_zarr_helper import save_ome_zarr
from src.image.util import *
from src.util import *


def create_source(filename):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        source = TiffSource(filename)
    elif ext.startswith('zar'):
        source = OmeZarrSource(filename)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    return source


def init_sims(files, flatfield_quantile=None, invert_x_coordinates=False, is_fix_missing_rotation=False):
    sims = []
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

    translations = []
    for source, image in zip(sources, images):
        translation = np.array(get_value_units_micrometer(source.position))
        if invert_x_coordinates:
            translation[0] = -translation[0]
        translation[1] = -translation[1]
        translations.append(translation)

    if is_fix_missing_rotation:
        source0 = sources[0]
        size = np.array(source0.get_size()) * source0.get_pixel_size_micrometer()
        translations = fix_missing_rotation(translations, size)

    #translations = [np.array(translation) * 1.25 for translation in translations]

    for source, image, translation in zip(sources, images, translations):
        # transform #dimensions need to match
        scale = convert_xyz_to_dict(source.get_pixel_size_micrometer())
        translation = convert_xyz_to_dict(translation)
        if not translation.keys() == scale.keys():
            translation = {key: translation.get(key, 0) for key in scale.keys()}
        channel_labels = [channel.get('label', '') for channel in source.get_channels()]
        sim = si_utils.get_sim_from_array(
            image,
            dims=list(output_order),
            scale=scale,
            translation=translation,
            transform_key="stage_metadata",
            c_coords=channel_labels
        )
        sims.append(sim)

    return sims


def fix_missing_rotation(positions0, size):
    # in [xy(z)]
    positions = []
    positions_centre = np.mean(positions0, 0)
    center_index = np.argmin([math.dist(position, positions_centre) for position in positions0])
    center_position = positions0[center_index]
    pairs = get_orthogonal_pairs_from_tiles(positions0, size)
    angles = []
    rotations = []
    for pair in pairs:
        vector = positions0[pair[1]] - positions0[pair[0]]
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
    for position0 in positions0:
        position = apply_transform([position0], transform)[0]
        positions.append(position)
    return positions


def normalise(sims, use_global=True):
    new_sims = []
    # global mean and stddev
    if use_global:
        means = []
        stddevs = []
        for sim in sims:
            means.append(np.mean(sim.data))
            stddevs.append(np.std(sim.data))
        mean = np.mean(means)
        stddev = np.mean(stddevs)
    else:
        mean = 0
        stddev = 1
    # normalise all images
    for sim in sims:
        if not use_global:
            mean = np.mean(sim.data)
            stddev = np.std(sim.data)
        image = float2int_image(np.clip((sim.data - mean) / stddev, 0, 1))
        new_sim = si_utils.get_sim_from_array(
            image,
            dims=sim.dims,
            scale=si_utils.get_spacing_from_sim(sim),
            translation=si_utils.get_origin_from_sim(sim),
            transform_key=si_utils.get_tranform_keys_from_sim(sim)[0],
            c_coords=sim.c
        )
        new_sims.append(new_sim)
    return new_sims


def get_orthogonal_pairs_from_tiles(origins, image_size_um):
    """
    Get pairs of orthogonal neighbors from a list of tiles.
    This assumes that the tiles are placed on a regular grid.
    """

    threshold = image_size_um
    threshold_half = threshold / 2

    # get pairs of neighboring tiles
    pairs = []
    for i, j in np.transpose(np.triu_indices(len(origins), 1)):
        relvec = abs(origins[i] - origins[j])
        if np.any(relvec < threshold_half) and np.all(relvec < threshold):
            pairs.append((i, j))

    return pairs


def register(sims0, reg_channel=None, reg_channel_index=None, normalisation=False, filter_foreground=False,
             use_orthogonal_pairs=False, use_rotation=False, channels=[]):
    if isinstance(reg_channel, int):
        reg_channel_index = reg_channel
        reg_channel = None

    is_channel_overlay = (len(channels) > 0)
    # normalisation
    if normalisation:
        sims = normalise(sims0, use_global=not is_channel_overlay)
    else:
        sims = sims0

    msims0 = [msi_utils.get_msim_from_sim(sim) for sim in sims0]
    msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]

    if filter_foreground:
        print('Filtering foreground tiles...')
        tile_vars = [np.asarray(np.std(sim)).item() for sim in sims]
        threshold = np.median(tile_vars)    # using median by definition 50% of the tiles
        foregrounds = (tile_vars > threshold)
        foreground_msims = [msim for msim, foreground in zip(msims, foregrounds) if foreground]
        #threshold, foregrounds = filter_noise_images(sims)
        #foreground_msims = [msim for msim, foreground in zip(msims, foregrounds) if foreground]
        print(f'Foreground tiles: {len(foreground_msims)} / {len(msims)}')

        # duplicate transform keys
        for msim0, msim in zip (msims0, msims):
            msi_utils.set_affine_transform(
                msim0,
                param_utils.identity_transform(ndim=2, t_coords=[0]),
                transform_key='registered',
                base_transform_key='stage_metadata')
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
        register_sims = [msi_utils.get_sim_from_msim(msim) for msim in register_msims]
        origins = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in register_sims])
        sim0 = register_sims[0]
        size = si_utils.get_shape_from_sim(sim0, asarray=True) * si_utils.get_spacing_from_sim(sim0, asarray=True)
        pairs = get_orthogonal_pairs_from_tiles(origins, size)
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
            if is_channel_overlay:
                pairwise_reg_func = registration.registration_ANTsPy
            else:
                pairwise_reg_func = registration.phase_correlation_registration
            mappings = registration.register(
                register_msims,
                reg_channel=reg_channel,
                reg_channel_index=reg_channel_index,
                transform_key="stage_metadata",
                new_transform_key="registered",
                pairs=pairs,
                pre_registration_pruning_method=None,
                pairwise_reg_func=pairwise_reg_func,
                plot_summary=True
            )

    progress.update()
    progress.close()
    mappings_dict = {int(index): mapping.data[0].tolist() for index, mapping in zip(indices, mappings)}
    distances = [np.linalg.norm(apply_transform([(0, 0)], np.array(mapping))[0]) for mapping in mappings_dict.values()]

    if is_channel_overlay:
        sim0 = sims0[0]
        spatial_dims = si_utils.get_spatial_dims_from_sim(sim0)
        size = [sim0.sizes[dim] * si_utils.get_spacing_from_sim(sim0)[dim] for dim in spatial_dims]
        norm_distance = np.sum(distances) / np.linalg.norm(size)
        score = 1 - min(math.sqrt(norm_distance), 1)
    else:
        # Coefficient of variation
        cv = np.std(distances) / np.mean(distances)
        score = 1 - min(cv / 10, 1)

    for msim, msim0 in zip(msims, msims0):
        msi_utils.set_affine_transform(
            msim0,
            msi_utils.get_transform_from_msim(msim, transform_key='registered'),
            transform_key='registered')

    print('Fusing...')
    # convert to multichannel images
    sims0 = [msi_utils.get_sim_from_msim(msim) for msim in msims0]
    if is_channel_overlay:
        output_stack_properties = si_utils.get_stack_properties_from_sim(sims0[0])
        channel_sims = [fusion.fuse(
            [sim],
            transform_key="registered",
            output_stack_properties=output_stack_properties
        ) for sim in sims0]
        channel_sims = [sim.assign_coords({'c': [channels[simi]['label']]})
                        for simi, sim in enumerate(channel_sims)]
        #fused_image = xr.combine_by_coords([sim.rename(None) for sim in channel_sims], combine_attrs='override')
        fused_image = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
    else:
        fused_image = fusion.fuse(
            sims0,
            transform_key="registered"
        )
    return mappings_dict, score, msims0, fused_image


def save_image(filename, data, transform_key=None, channels=None, positions=None,
               npyramid_add=4, pyramid_downsample=2, out_params={}):
    dimension_order = ''.join(data.dims)
    sdims = si_utils.get_spatial_dims_from_sim(data)
    nsdims = si_utils.get_nonspatial_dims_from_sim(data)
    nchannels = data.sizes.get('c', 1)

    pixel_size = [si_utils.get_spacing_from_sim(data)[dim] for dim in sdims]

    origin = si_utils.get_origin_from_sim(data)
    if transform_key is not None:
        transform = si_utils.get_affine_from_sim(data, transform_key)
        for nsdim in nsdims:
            if nsdim in transform.dims:
                transform = transform.sel(
                    {
                        nsdim: transform.coords[nsdim][0]
                        for nsdim in transform.dims
                    }
                )
        transform = np.array(transform)
        transform_translation = param_utils.translation_from_affine(transform)
        for isdim, sdim in enumerate(sdims):
            origin[sdim] += transform_translation[isdim]
    position = [origin[dim] for dim in sdims]
    if positions is None:
        positions = [position] * nchannels

    if channels is None:
        channels = data.attrs.get('channels', [])

    npyramid_add = get_max_downsamples(data.shape, npyramid_add, pyramid_downsample)
    scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)

    if 'format' in out_params and 'zar' in out_params['format']:
        print('writing ome-zarr')
        save_ome_zarr(filename + '.ome.zarr', data.data, dimension_order, pixel_size, channels, position, scaler=scaler)
    if 'format' in out_params and 'tif' in out_params['format']:
        print('writing ome-tiff')
        save_ome_tiff(filename + '.ome.tiff', data.data, pixel_size, channels, positions, scaler=scaler)


def dir_regex(pattern):
    dir = os.path.dirname(pattern)
    file_pattern = os.path.basename(pattern)
    files = [os.path.join(dir, file) for file in os.listdir(dir) if re.search(file_pattern, file)]
    files_sorted = sorted(files, key=lambda file: find_all_numbers(get_filetitle(file)))
    return files_sorted


def run_stitch(input, target, params):
    reg_params = params['registration']
    out_params = params['output']

    invert_x_coordinates = reg_params.get('invert_x_coordinates', False)
    flatfield_quantile = reg_params.get('flatfield_quantile')
    normalisation = reg_params.get('normalisation', False)
    filter_foreground = reg_params.get('filter_foreground', False)
    use_orthogonal_pairs = reg_params.get('use_orthogonal_pairs', False)
    is_fix_missing_rotation = reg_params.get('fix_missing_rotation', False)
    use_rotation = reg_params.get('use_rotation', False)
    reg_channel = reg_params.get('reg_channel', 0)

    show_original = out_params.get('show_original', False)
    npyramid_add = out_params.get('npyramid_add', 0)
    pyramid_downsample = out_params.get('pyramid_downsample', 2)
    channels = out_params.get('channels', [])

    original_positions_filename = target + 'positions_original.png'
    original_fused_filename = target + 'original'
    registered_positions_filename = target + 'positions_registered.png'
    registered_fused_filename = target + 'registered'

    mvsr_logger = logging.getLogger('multiview_stitcher.registration')
    mvsr_logger.setLevel(logging.INFO)
    if len(mvsr_logger.handlers) == 0:
        mvsr_logger.addHandler(logging.StreamHandler())

    if isinstance(input, list):
        filenames = input
        file_indices = list(range(len(filenames)))
    else:
        filenames = dir_regex(input)
        file_indices = ['-'.join(map(str, find_all_numbers(get_filetitle(filename))[-2:])) for filename in filenames]

    if len(filenames) <= 1:
        print('Skipping #tiles <= 1')
        return

    print('Initialising tiles...')
    sims = init_sims(filenames, flatfield_quantile=flatfield_quantile, invert_x_coordinates=invert_x_coordinates,
                     is_fix_missing_rotation=is_fix_missing_rotation)

    if show_original:
        # before registration:
        print('Fusing original...')
        original_fused = fusion.fuse(
            sims,
            transform_key='stage_metadata'
        )

        # plot the tile configuration
        print('Plotting tiles...')
        msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]
        vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False,
                                 view_labels=file_indices, view_labels_size=3,
                                 show_plot=False, output_filename=original_positions_filename)

        print('Saving fused image...')
        save_image(original_fused_filename, original_fused, transform_key='stage_metadata',
                   npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample, out_params=out_params)

    mappings, score, msims, registered_fused = (
        register(sims, reg_channel, normalisation=normalisation, filter_foreground=filter_foreground,
                 use_orthogonal_pairs=use_orthogonal_pairs, use_rotation=use_rotation))
    print(f'Score: {score:.3f}')
    mappings2 = {get_filetitle(filenames[index]): mapping for index, mapping in mappings.items()}
    with open(target + 'mappings.json', 'w') as file:
        json.dump(mappings2, file, indent=4)

    # plot the tile configuration after registration
    print('Plotting tiles...')
    vis_utils.plot_positions(msims, transform_key='registered', use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=False, output_filename=registered_positions_filename)

    print('Saving fused image...')
    positions = [apply_transform([(0, 0)], np.array(mapping))[0] for mapping in mappings.values()]
    save_image(registered_fused_filename, registered_fused, transform_key='registered', positions=positions,
               npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample, out_params=out_params)


def run_stitch_overlay():
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    channels = [{'label': 'Reflection', 'color': (1, 1, 1)},
                {'label': 'Fluorescence', 'color': (0, 1, 0)}]

    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/subselection/tiles_1_MMStack_New Grid 1-Grid_(?!0_0.ome.tif).*'     # 3x3 subselection
    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    input = '/nemo/project/proj-czi-vp/raw/lm/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    filenames = dir_regex(input)
    sims1 = init_sims(filenames, flatfield_quantile=0.95, invert_x_coordinates=True)
    mappings1, score, msims1, registered_fused1 = (
        register(sims1, 0, filter_foreground=True, use_orthogonal_pairs=True))
    print(f'Score: {score:.3f}')

    print('Plotting tiles...')
    vis_utils.plot_positions(msims1, transform_key='registered', use_positional_colors=False,
                             show_plot=False, output_filename=os.path.join(output_dir, 'tiles_registered1.png'))

    print('Saving fused image...')
    save_image(os.path.join(output_dir, 'registered1'), registered_fused1, transform_key='registered')

    #input = 'D:/slides/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/subselection/tiles_1_MMStack_New Grid 1-Grid_(?!0_0.ome.tif).*'  # 3x3 subselection
    #input = 'D:/slides/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/EM04768_01_sub_04_fluorescence_10x/converted/.*.ome.tif'
    input = '/nemo/project/proj-czi-vp/raw/lm/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/EM04768_01_sub_04_fluorescence_10x/converted/.*.ome.tif'
    filenames = dir_regex(input)
    sims2 = init_sims(filenames, flatfield_quantile=0.95, invert_x_coordinates=True)
    mappings2, score, msims2, registered_fused2 = (
        register(sims2, 0, filter_foreground=True, use_orthogonal_pairs=True))
    print(f'Score: {score:.3f}')

    print('Plotting tiles...')
    vis_utils.plot_positions(msims2, transform_key='registered', use_positional_colors=False,
                             show_plot=False, output_filename=os.path.join(output_dir, 'tiles_registered2.png'))

    print('Saving fused image...')
    save_image(os.path.join(output_dir, 'registered2'), registered_fused2, transform_key='registered')

    sims = [registered_fused1, registered_fused2]
    # set dummy position
    for sim in sims:
        si_utils.set_sim_affine(sim,
                                param_utils.identity_transform(ndim=2, t_coords=[0]),
                                transform_key='stage_metadata')
    mappings, score, msims, registered_fused =(
        register(sims, 0, normalisation=True, filter_foreground=False, use_orthogonal_pairs=False,
                 channels=channels))
    print(f'Score: {score:.3f}')

    with open(os.path.join(output_dir, 'mappings_overlay.json'), 'w') as file:
        json.dump(mappings, file, indent=4)

    print('Plotting overlay...')
    vis_utils.plot_positions(msims, transform_key='registered', use_positional_colors=False,
                             show_plot=False, output_filename=os.path.join(output_dir, 'overlay_registered.png'))

    print('Saving fused image...')
    save_image(os.path.join(output_dir, 'registered'), registered_fused, transform_key='registered', channels=channels)


def run(params):
    print(f'Multiview-stitcher Version: {multiview_stitcher.__version__}')

    sources = ensure_list(params['input']['source'])
    break_on_error = params['output']['break_on_error']

    for source in sources:
        print('Source:', source)
        try:
            source_dir = os.path.dirname(source)
            target = os.path.join(source_dir, params['output']['target'])
            target_dir = os.path.dirname(target)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            run_stitch(source, target, params)
        except Exception as e:
            print(f'Error: {e}')
            if break_on_error:
                break

    print('Done!')
    print()


if __name__ == '__main__':
    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/subselection/tiles_1_MMStack_New Grid 1-Grid_(?!0_0.ome.tif).*'     # 3x3 subselection
    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/tiles_1_MMStack_New Grid 1-Grid_5_.*.ome.tif'     # one column of tiles
    #input = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    #input = 'D:/slides/EM04768_01_substrate_04/Fluorescence/20_percent_overlap/EM04768_01_sub_04_fluorescence_10x/converted/.*.ome.tif'
    input = '/nemo/project/proj-czi-vp/raw/lm/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'

    #input = ['output_reflect_orth/registered.ome.zarr', 'output_fluor_orth/registered.ome.zarr']

    #input = 'D:/slides/EM04768_01_substrate_04/EM/a0004/roi0000/t.*/.*.ome.tif'
    #input = 'D:/slides/EM04768_01_substrate_04/EM/a0004/roi0000/.*.ome.tif'

    output_dir = 'output'

    run_stitch(input, output_dir)
    #run_stitch_overlay()
