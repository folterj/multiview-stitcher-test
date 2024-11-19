# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

from contextlib import nullcontext
from dask.diagnostics import ProgressBar
import json
import logging
import math
import multiview_stitcher
from multiview_stitcher import registration, fusion, msi_utils, vis_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
import numpy as np
import os
from ome_zarr.scale import Scaler
import re
import shutil
from tqdm import tqdm
import xarray as xr

from src.OmeZarrSource import OmeZarrSource
from src.TiffSource import TiffSource
from src.image.ome_tiff_helper import save_ome_tiff
from src.image.ome_zarr_helper import save_ome_zarr
from src.image.util import *
from src.util import *


def init_logging(params_general):
    verbose = params_general.get('verbose', False)
    log_filename = params_general.get('log_filename', 'logfile.log')
    log_format = params_general.get('log_format')
    basepath = os.path.dirname(log_filename)
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    handlers = [logging.FileHandler(log_filename, encoding='utf-8')]
    if verbose:
        handlers += [logging.StreamHandler()]
        # expose multiview_stitcher.registration logger and make more verbose
        mvsr_logger = logging.getLogger('multiview_stitcher.registration')
        mvsr_logger.setLevel(logging.INFO)
        if len(mvsr_logger.handlers) == 0:
            mvsr_logger.addHandler(logging.StreamHandler())

    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, encoding='utf-8')


def create_source(filename):
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    if ext.startswith('tif'):
        source = TiffSource(filename)
    elif ext.startswith('zar'):
        source = OmeZarrSource(filename)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    return source


def init_tiles(files, flatfield_quantile=None, invert_x_coordinates=False, is_fix_missing_rotation=False,
               verbose=False):
    sims = []
    sources = [create_source(file) for file in files]
    nchannels = sources[0].get_nchannels()
    images = []
    logging.info('Init tiles...')
    for source in tqdm(sources, disable=not verbose):
        output_order = 'yx'
        if source.get_nchannels() > 1:
            output_order += 'c'
        image = redimension_data(source.get_source_dask()[0],
                                 source.dimension_order, output_order)
        images.append(image)

    if flatfield_quantile is not None:
        logging.info('Applying flatfield correction...')
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
    pairs, angles = get_orthogonal_pairs_from_tiles(positions0, size)
    if len(pairs) > 0:
        rotation = np.mean(angles)
        transform = create_transform(center=center_position, angle=rotation)
        for position0 in positions0:
            position = apply_transform([position0], transform)[0]
            positions.append(position)
    else:
        positions = positions0
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
            transform_key='stage_metadata',
            c_coords=sim.c
        )
        new_sims.append(new_sim)
    return new_sims


def get_orthogonal_pairs_from_tiles(origins, image_size_um):
    """
    Get pairs of orthogonal neighbors from a list of tiles.
    Tiles don't have to be placed on a regular grid.
    """
    pairs = []
    angles = []
    for i, j in np.transpose(np.triu_indices(len(origins), 1)):
        origini = origins[i]
        originj = origins[j]
        distance = math.dist(origini, originj)
        if distance < max(image_size_um):
            pairs.append((i, j))
            vector = origini - originj
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            if distance < min(image_size_um):
                angle += 90
            while angle < -90:
                angle += 180
            while angle > 90:
                angle -= 180
            angles.append(angle)
    return pairs, angles


def register(sims0, method, reg_channel=None, reg_channel_index=None, normalisation=False, filter_foreground=False,
             use_orthogonal_pairs=False, use_rotation=False, channels=[], verbose=False):
    if isinstance(reg_channel, int):
        reg_channel_index = reg_channel
        reg_channel = None

    is_channel_overlay = (len(channels) > 0)
    # normalisation
    if normalisation:
        use_global = not is_channel_overlay
        if use_global:
            logging.info('Normalising tiles (global)...')
        else:
            logging.info('Normalising tiles...')
        sims = normalise(sims0, use_global=use_global)
    else:
        sims = sims0

    msims0 = [msi_utils.get_msim_from_sim(sim) for sim in sims0]
    msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]

    if filter_foreground:
        logging.info('Filtering foreground tiles...')
        tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
        threshold1 = np.mean(tile_vars)
        threshold2 = np.median(tile_vars)
        threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
        threshold = min(threshold1, threshold2, threshold3)
        foregrounds = (tile_vars >= threshold)
        foreground_msims = [msim for msim, foreground in zip(msims, foregrounds) if foreground]
        logging.info(f'Foreground tiles: {len(foreground_msims)} / {len(msims)}')

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

    logging.info('Registering...')
    if verbose:
        progress = tqdm()

    if use_orthogonal_pairs:
        register_sims = [msi_utils.get_sim_from_msim(msim) for msim in register_msims]
        origins = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in register_sims])
        sim0 = register_sims[0]
        size = si_utils.get_shape_from_sim(sim0, asarray=True) * si_utils.get_spacing_from_sim(sim0, asarray=True)
        pairs, _ = get_orthogonal_pairs_from_tiles(origins, size)
        logging.info(f'#pairs: {len(pairs)}')
    else:
        pairs = None
    with ProgressBar() if verbose else nullcontext():
        try:
            if use_rotation:
                # phase shift registration
                mappings1, df1 = registration.register(
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
                    plot_summary=True,
                    return_metrics=True
                )
                # affine registration
                mappings2, df2 = registration.register(
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
                    plot_summary=True,
                    return_metrics=True
                )
                mappings = mappings2
                df = df2
            else:
                if 'ant' in method:
                    pairwise_reg_func = registration.registration_ANTsPy
                else:
                    pairwise_reg_func = registration.phase_correlation_registration
                logging.info(f'Registration method: {pairwise_reg_func.__name__}')

                #abs_tol = 5 * np.max([np.max(si_utils.get_spacing_from_sim(sim, asarray=True)) for sim in sims])

                mappings, df = registration.register(
                    register_msims,
                    reg_channel=reg_channel,
                    reg_channel_index=reg_channel_index,
                    transform_key="stage_metadata",
                    new_transform_key="registered",
                    pairs=pairs,
                    pre_registration_pruning_method=None,
                    pairwise_reg_func=pairwise_reg_func,

                    #registration_binning={dim: 1 for dim in 'yx'},
                    #groupwise_resolution_method='shortest_paths',

                    #groupwise_resolution_kwargs={
                    #    'transform': 'translation',
                    #    'max_residual_max_mean_ratio': 3.,
                    #    'abs_tol': abs_tol,
                    #},

                    post_registration_do_quality_filter=True,
                    post_registration_quality_threshold=0.1,

                    plot_summary=True,
                    return_metrics=True
                )

            final_residual = list(df['mean_residual'])[-1]

            mappings_dict = {index: mapping.data[0].tolist() for index, mapping in zip(indices, mappings)}
            distances = [np.linalg.norm(apply_transform([(0, 0)], np.array(mapping))[0]) for mapping in
                         mappings_dict.values()]

            if is_channel_overlay:
                sim0 = sims0[0]
                spatial_dims = si_utils.get_spatial_dims_from_sim(sim0)
                size = [sim0.sizes[dim] * si_utils.get_spacing_from_sim(sim0)[dim] for dim in spatial_dims]
                norm_distance = np.sum(distances) / np.linalg.norm(size)
                confidence = 1 - min(math.sqrt(norm_distance), 1)
            else:
                # Coefficient of variation
                cvar = np.std(distances) / np.mean(distances)
                confidence = 1 - min(cvar / 10, 1)

            for msim, msim0 in zip(msims, msims0):
                msi_utils.set_affine_transform(
                    msim0,
                    msi_utils.get_transform_from_msim(msim, transform_key='registered'),
                    transform_key='registered')

        except NotEnoughOverlapError:
            final_residual = 0
            confidence = 0
            for msim0 in msims0:
                msi_utils.set_affine_transform(
                    msim0,
                    param_utils.identity_transform(ndim=2, t_coords=[0]),
                    transform_key='registered',
                    base_transform_key='stage_metadata')
            mappings_dict = {index: np.eye(3).tolist() for index, _ in enumerate(msims0)}

    if verbose:
        progress.update()
        progress.close()

    logging.info('Fusing...')
    # convert to multichannel images
    sims0 = [msi_utils.get_sim_from_msim(msim) for msim in msims0]
    if is_channel_overlay:
        output_stack_properties = si_utils.get_stack_properties_from_sim(sims0[0])
        channel_sims = [fusion.fuse(
            [sim],
            transform_key='registered',
            output_stack_properties=output_stack_properties
        ) for sim in sims0]
        channel_sims = [sim.assign_coords({'c': [channels[simi]['label']]})
                        for simi, sim in enumerate(channel_sims)]
        #fused_image = xr.combine_by_coords([sim.rename(None) for sim in channel_sims], combine_attrs='override')
        fused_image = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
    else:
        fused_image = fusion.fuse(
            sims0,
            transform_key='registered'
        )
    return {'mappings': mappings_dict,
            'final_residual': final_residual,
            'confidence': confidence,
            'msims': msims0,
            'fused_image': fused_image}


def save_image(filename, data, transform_key=None, channels=None, positions=None,
               npyramid_add=4, pyramid_downsample=2, params={}):
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

    if 'zar' in params.get('format', 'zar'):
        logging.info('Saving OME-Zarr...')
        save_ome_zarr(filename + '.ome.zarr', data.data, dimension_order, pixel_size, channels, position, scaler=scaler)
    if 'tif' in params.get('format', 'tif'):
        logging.info('Saving OME-Tiff...')
        save_ome_tiff(filename + '.ome.tiff', data.data, pixel_size, channels, positions, scaler=scaler)


def run_operation(params, params_general):
    input = params['input']
    output_params = params_general.get('output', {})
    method = params.get('method', '').lower()
    invert_x_coordinates = params.get('invert_x_coordinates', False)
    flatfield_quantile = params.get('flatfield_quantile')
    normalisation = params.get('normalisation', False)
    filter_foreground = params.get('filter_foreground', False)
    use_orthogonal_pairs = params.get('use_orthogonal_pairs', False)
    is_fix_missing_rotation = params.get('fix_missing_rotation', False)
    use_rotation = params.get('use_rotation', False)
    reg_channel = params.get('channel', 0)
    channels = params.get('extra_metadata', {}).get('channels', [])
    clear = params_general['output'].get('clear', False)

    show_original = params_general.get('show_original', False)
    npyramid_add = params_general.get('npyramid_add', 0)
    pyramid_downsample = params_general.get('pyramid_downsample', 2)
    verbose = params_general.get('verbose', False)

    if isinstance(input, list):
        filenames = input
        file_indices = list(range(len(filenames)))
    else:
        filenames = dir_regex(input)
        file_indices = ['-'.join(map(str, find_all_numbers(get_filetitle(filename))[-2:])) for filename in filenames]

    if len(filenames) == 0:
        logging.warning('Skipping (no tiles)')
        return

    input_dir, _ = split_path(ensure_list(input)[0])
    output = os.path.join(input_dir, params['output'])
    output_dir = os.path.dirname(output)
    if clear:
        shutil.rmtree(output_dir, ignore_errors=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_positions_filename = output + 'positions_original.png'
    original_fused_filename = output + 'original'
    registered_positions_filename = output + 'positions_registered.png'
    registered_fused_filename = output + 'registered'

    logging.info('Initialising tiles...')
    sims = init_tiles(filenames, flatfield_quantile=flatfield_quantile, invert_x_coordinates=invert_x_coordinates,
                      is_fix_missing_rotation=is_fix_missing_rotation, verbose=verbose)

    if len(filenames) == 1:
        logging.warning('Skipping registration (single tile)')
        save_image(registered_fused_filename, sims[0], channels=channels,
                   npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample, params=params_general['output'])
        return

    if show_original:
        # before registration:
        logging.info('Fusing original...')
        original_fused = fusion.fuse(
            sims,
            transform_key='stage_metadata'
        )

        # plot the tile configuration
        logging.info('Plotting tiles...')
        msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]
        vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False,
                                 view_labels=file_indices, view_labels_size=3,
                                 show_plot=False, output_filename=original_positions_filename)

        logging.info('Saving fused image...')
        save_image(original_fused_filename, original_fused, transform_key='stage_metadata',
                   npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample, params=output_params)

    results = register(sims, method, reg_channel, normalisation=normalisation, filter_foreground=filter_foreground,
                       use_orthogonal_pairs=use_orthogonal_pairs, use_rotation=use_rotation, channels=channels,
                       verbose=verbose)
    logging.info(f'Final residual: {results["final_residual"]:.3f} Confidence: {results["confidence"]:.3f}')
    mappings = results['mappings']
    mappings2 = {get_filetitle(filenames[index]): mapping for index, mapping in mappings.items()}
    with open(output + 'mappings.json', 'w') as file:
        json.dump(mappings2, file, indent=4)

    # plot the tile configuration after registration
    logging.info('Plotting tiles...')
    vis_utils.plot_positions(results['msims'], transform_key='registered', use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=False, output_filename=registered_positions_filename)

    logging.info('Saving fused image...')
    positions = [apply_transform([(0, 0)], np.array(mapping))[0] for mapping in mappings.values()]
    save_image(registered_fused_filename, results['fused_image'], transform_key='registered',
               channels=channels, positions=positions,
               npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample, params=params_general['output'])


def run(params):
    logging.info(f'Multiview-stitcher Version: {multiview_stitcher.__version__}')

    params_general = params['general']
    break_on_error = params_general.get('break_on_error', False)

    for operation in tqdm(params['operations']):
        input = operation['input']
        logging.info(f'Input: {input}')
        try:
            input_dir, _ = split_path(ensure_list(input)[0])
            if os.path.exists(input_dir):
                run_operation(operation, params_general)
            else:
                raise FileNotFoundError(f'Input directory not found: {input_dir}')
        except Exception as e:
            logging.exception(f'Error processing: {input}')
            print(f'Error processing: {input}: {e}')
            if break_on_error:
                break

    logging.info('Done!')
