# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

import csv
from contextlib import nullcontext
import cv2 as cv
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
import shutil
from sklearn.neighbors import KDTree
from tqdm import tqdm
import xarray as xr

from src.OmeZarrSource import OmeZarrSource
from src.TiffSource import TiffSource
from src.Video import Video
from src.image.ome_helper import save_image
from src.image.ome_tiff_helper import load_tiff, save_tiff
from src.image.util import *
from src.util import *


ndims = 3
pixel_size_xyz = [1, 1, 1]


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


def init_tiles(files, transform_key, d3=False,
               invert_x_coordinates=False, normalise_orientation=False, reset_coordinates=False,
               verbose=False):
    sims = []
    sources = [create_source(file) for file in files]
    images = []
    logging.info('Init tiles...')
    for source in tqdm(sources, disable=not verbose):
        output_order = 'zyx' if d3 else 'yx'
        if source.get_nchannels() > 1:
            output_order += 'c'
        image = redimension_data(source.get_source_dask()[0],
                                 source.dimension_order, output_order)
        images.append(image)

    translations = []
    rotations = []
    for source, image in zip(sources, images):
        if reset_coordinates or len(source.get_position()) == 0:
            translation = np.zeros(3)
        else:
            translation = np.array(source.get_position_micrometer())
            if invert_x_coordinates:
                translation[0] = -translation[0]
                translation[1] = -translation[1]
        translations.append(translation)
        rotations.append(source.get_rotation())

    if normalise_orientation:
        source0 = sources[0]
        size = np.array(source0.get_size()) * source0.get_pixel_size_micrometer()
        translations, rotations = normalise_rotated_positions(translations, rotations, size)

    #translations = [np.array(translation) * 1.25 for translation in translations]

    for source, image, translation, rotation in zip(sources, images, translations, rotations):
        # transform #dimensions need to match
        scale_dict = convert_xyz_to_dict(source.get_pixel_size_micrometer())
        if len(scale_dict) > 0 and 'z' not in scale_dict:
            scale_dict['z'] = 1
        translation_dict = convert_xyz_to_dict(translation)
        if len(translation_dict) > 0 and 'z' not in translation_dict:
            translation_dict['z'] = 0
        channel_labels = [channel.get('label', '') for channel in source.get_channels()]
        if rotation is None or normalise_orientation:
            transform = None
        else:
            transform = param_utils.invert_coordinate_order(create_transform(translation, rotation))
        sim = si_utils.get_sim_from_array(
            image,
            dims=list(output_order),
            scale=scale_dict,
            translation=translation_dict,
            affine=transform,
            transform_key=transform_key,
            c_coords=channel_labels
        )
        sims.append(sim)

    return sims, translations, rotations


def normalise_rotated_positions(positions0, rotations0, size):
    # in [xy(z)]
    positions = []
    rotations = []
    positions_centre = np.mean(positions0, 0)
    center_index = np.argmin([math.dist(position, positions_centre) for position in positions0])
    center_position = positions0[center_index]
    pairs, angles = get_orthogonal_pairs_from_tiles(positions0, size)
    if len(pairs) > 0:
        mean_angle = np.mean(angles)
        for position0, rotation in zip(positions0, rotations0):
            if rotation is None:
                rotation = -mean_angle
            transform = create_transform(center=center_position, angle=-rotation)
            position = apply_transform([position0], transform)[0]
            positions.append(position)
            rotations.append(rotation)
    else:
        positions = positions0
        rotations = rotations0
    return positions, rotations


def calc_foreground_map(sims):
    if len(sims) <= 2:
        return [True] * len(sims)
    sims = [sim.squeeze().astype(np.float32) for sim in sims]
    median_image = calc_images_median(sims).astype(np.float32)
    difs = [np.mean(np.abs(sim - median_image), (0, 1)) for sim in sims]
    # or use stddev instead of mean?
    threshold = np.mean(difs, 0)
    #threshold, _ = cv.threshold(np.array(difs).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
    #threshold, foregrounds = filter_noise_images(channel_images)
    map = (difs > threshold)
    if np.all(map == False):
        return [True] * len(sims)
    return map


def flatfield_correction(sims, transform_key, foreground_map, flatfield_quantile):
    new_sims = []
    dtype = sims[0].dtype
    norm_image_filename = f'resources/norm{flatfield_quantile}.tiff'
    if os.path.exists(norm_image_filename):
        logging.warning('Loading cached normalisation image')
        max_image = load_tiff(norm_image_filename)
    else:
        back_sims = [sim for sim, is_foreground in zip(sims, foreground_map) if not is_foreground]
        norm_images = create_quantile_images(back_sims, quantiles=[flatfield_quantile])
        max_image = norm_images[0]
        maxval = 2 ** (8 * dtype.itemsize) - 1
        max_image = max_image / np.float32(maxval)
        save_tiff(norm_image_filename, max_image)
    for sim in sims:
        image = float2int_image(image_flatfield_correction(int2float_image(sim), bright=max_image), dtype)
        new_sim = si_utils.get_sim_from_array(
            image,
            dims=sim.dims,
            scale=si_utils.get_spacing_from_sim(sim),
            translation=si_utils.get_origin_from_sim(sim),
            transform_key=transform_key,
            affine=si_utils.get_affine_from_sim(sim, transform_key),
            c_coords=sim.c
        )
        new_sims.append(new_sim)
    return new_sims


def normalise(sims, transform_key, use_global=True):
    new_sims = []
    # global mean and stddev
    if use_global:
        mins = []
        ranges = []
        for sim in sims:
            min = np.mean(sim)
            range = np.std(sim)
            #min, max = get_image_window(sim, low=0.01, high=0.99)
            #range = max - min
            mins.append(min)
            ranges.append(range)
        min = np.mean(mins)
        range = np.mean(ranges)
    else:
        min = 0
        range = 1
    # normalise all images
    for sim in sims:
        if not use_global:
            min = np.mean(sim)
            range = np.std(sim)
        image = float2int_image(np.clip((sim - min) / range, 0, 1))
        new_sim = si_utils.get_sim_from_array(
            image,
            dims=sim.dims,
            scale=si_utils.get_spacing_from_sim(sim),
            translation=si_utils.get_origin_from_sim(sim),
            transform_key=transform_key,
            affine=si_utils.get_affine_from_sim(sim, transform_key),
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


def pairwise_registration_dummy(
    fixed_data, moving_data,
    **kwargs, # additional keyword arguments passed `pairwise_reg_func_kwargs`
    ) -> dict:
    transform=cv.getRotationMatrix2D((fixed_data.shape[0]//2, fixed_data.shape[1]//2), 38, 1)
    transform = np.vstack([transform, [0, 0, 1]])
    transform[:, 2] += [300, 25, 0]

    return {
        "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
        "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
    }


def detect_features(data):
    feature_model = cv.ORB_create()
    kp, desc = feature_model.detectAndCompute(data, None)
    points = [kp1.pt for kp1 in kp]
    if len(points) >= 2:
        tree = KDTree(points, leaf_size=2)
        dist, ind = tree.query(points, k=2)
        nn_distance = np.median(dist[:, 1])
    else:
        nn_distance = 1
    #image = cv.drawKeypoints(data, kp, data)
    #show_image(image)
    return points, desc, nn_distance


def pairwise_registration_features(
    fixed_data, moving_data,
    **kwargs, # additional keyword arguments passed `pairwise_reg_func_kwargs`
    ) -> dict:

    fixed_points, fixed_desc, nn_distance1 = detect_features(fixed_data.data)
    moving_points, moving_desc, nn_distance2 = detect_features(moving_data.data)
    nn_distance = np.mean([nn_distance1, nn_distance2])

    matcher = cv.BFMatcher()

    #matches = matcher.match(fixed_desc, moving_desc)
    matches0 = matcher.knnMatch(fixed_desc, moving_desc, k=2)
    matches = []
    for m, n in matches0:
        if m.distance < 0.75 * n.distance:
            matches.append(m)

    if len(matches) >= 4:
        fixed_points2 = np.float32([fixed_points[m.queryIdx] for m in matches])
        moving_points2 = np.float32([moving_points[m.trainIdx] for m in matches])
        transform, mask = cv.findHomography(fixed_points2, moving_points2, method=cv.USAC_MAGSAC, ransacReprojThreshold=nn_distance)
    else:
        logging.error('Not enough matches for feature-based registration')
        transform = np.eye(3)

    return {
        "affine_matrix": transform, # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
        "quality": 1 # float between 0 and 1 (if not available, set to 1.0)
    }


def pairwise_registration_cpd(
    fixed_data, moving_data,
    **kwargs, # additional keyword arguments passed `pairwise_reg_func_kwargs`
    ) -> dict:
    from probreg import cpd

    max_iter = kwargs.get('max_iter', 1000)

    fixed_points = points_to_3d([point for point, area in detect_area_points(fixed_data.data)])
    moving_points = points_to_3d([point for point, area in detect_area_points(moving_data.data)])

    if len(moving_points) > 1 and len(fixed_points) > 1:
        result_cpd = cpd.registration_cpd(moving_points, fixed_points, maxiter=max_iter)
        transformation = result_cpd.transformation
        S = transformation.scale * np.eye(3)
        R = transformation.rot
        T = np.eye(3) + np.hstack([np.zeros((3, 2)), transformation.t.reshape(-1, 1)])
        transform = T @ R @ S
    else:
        logging.error('Not enough points for CPD registration')
        transform = np.eye(3)

    return {
        "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
        "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
    }


def register(sims0, source_transform_key, reg_transform_key, params, params_general):
    global ndims, pixel_size_xyz
    sim0 = sims0[0]
    ndims = si_utils.get_ndim_from_sim(sim0)
    pixel_size_xyz = [si_utils.get_spacing_from_sim(sim0).get(dim, 1) for dim in 'xyz']

    operation = params['operation']
    method = params.get('method', '').lower()
    reg_channel = params.get('channel', 0)
    if isinstance(reg_channel, int):
        reg_channel_index = reg_channel
        reg_channel = None
    else:
        reg_channel_index = None

    flatfield_quantile = params.get('flatfield_quantile')
    normalisation = params.get('normalisation', False)
    filter_foreground = params.get('filter_foreground', False)
    use_orthogonal_pairs = params.get('use_orthogonal_pairs', False)
    use_rotation = params.get('use_rotation', False)
    show_filtered = params_general.get('show_filtered', False)
    verbose = params_general.get('verbose', False)

    extra_metadata = params.get('extra_metadata', {})
    channels = extra_metadata.get('channels', [])
    z_scale = extra_metadata.get('scale', {}).get('z', 1)
    is_stack = ('stack' in operation)
    is_channel_overlay = (len(channels) > 1)

    foreground_map = calc_foreground_map(sims0)
    if flatfield_quantile is not None:
        sims0 = flatfield_correction(sims0, source_transform_key, foreground_map, flatfield_quantile)

    if normalisation:
        use_global = not is_channel_overlay
        if use_global:
            logging.info('Normalising tiles (global)...')
        else:
            logging.info('Normalising tiles...')
        sims = normalise(sims0, source_transform_key, use_global=use_global)
    else:
        sims = sims0

    if filter_foreground:
        logging.info('Filtering foreground tiles...')
        #tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
        #threshold1 = np.mean(tile_vars)
        #threshold2 = np.median(tile_vars)
        #threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
        #threshold = min(threshold1, threshold2, threshold3)
        #foregrounds = (tile_vars >= threshold)
        foreground_sims = [sim for sim, is_foreground in zip(sims, foreground_map) if is_foreground]

        if show_filtered:
            filtered_filename = params['output'] + 'filtered.png'
            vis_utils.plot_positions(foreground_sims, transform_key=source_transform_key, use_positional_colors=False,
                                     view_labels_size=3, show_plot=verbose, output_filename=filtered_filename)

        logging.info(f'Foreground tiles: {len(foreground_sims)} / {len(sims)}')

        indices = np.where(foreground_map)[0]
        register_sims = foreground_sims
    else:
        indices = range(len(sims))
        register_sims = sims

    logging.info('Registering...')
    if verbose:
        progress = tqdm()

    if is_stack:
        register_sims = [si_utils.max_project_sim(sim, dim='z') for sim in register_sims]
        pairs = [(index, index + 1) for index in range(len(register_sims) - 1)]
    elif use_orthogonal_pairs:
        origins = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in register_sims])
        sim0 = register_sims[0]
        size = si_utils.get_shape_from_sim(sim0, asarray=True) * si_utils.get_spacing_from_sim(sim0, asarray=True)
        pairs, _ = get_orthogonal_pairs_from_tiles(origins, size)
        logging.info(f'#pairs: {len(pairs)}')
    else:
        pairs = None

    # copy source to reg transform: for background sims skipped in registration
    for sim in sims0:
        si_utils.set_sim_affine(
            sim,
            param_utils.identity_transform(ndim=ndims, t_coords=[0]),
            transform_key=reg_transform_key,
            base_transform_key=source_transform_key)

    if 'dummy' in method:
        pairwise_reg_func = pairwise_registration_dummy
    elif 'feature' in method:
        pairwise_reg_func = pairwise_registration_features
    elif 'cpd' in method:
        pairwise_reg_func = pairwise_registration_cpd
    elif 'ant' in method:
        pairwise_reg_func = registration.registration_ANTsPy
    else:
        pairwise_reg_func = registration.phase_correlation_registration
    logging.info(f'Registration method: {pairwise_reg_func.__name__}')

    if use_rotation:
        pairwise_reg_func_kwargs = {
            'transform_types': ['Rigid'],
            "aff_random_sampling_rate": 0.5,
            "aff_iterations": (2000, 2000, 1000, 1000),
            "aff_smoothing_sigmas": (4, 2, 1, 0),
            "aff_shrink_factors": (16, 8, 2, 1),
        }
        # these are the parameters for the groupwise registration (global optimization)
        groupwise_resolution_kwargs = {
            'transform': 'rigid',  # options include 'translation', 'rigid', 'affine'
        }
    else:
        pairwise_reg_func_kwargs = None
        groupwise_resolution_kwargs = None

    register_msims = [msi_utils.get_msim_from_sim(sim) for sim in register_sims]
    with ProgressBar() if verbose else nullcontext():
        try:
            mappings, metrics = registration.register(
                register_msims,
                reg_channel=reg_channel,
                reg_channel_index=reg_channel_index,
                transform_key=source_transform_key,
                new_transform_key=reg_transform_key,

                pairs=pairs,
                pre_registration_pruning_method=None,

                pairwise_reg_func=pairwise_reg_func,
                pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                groupwise_resolution_kwargs=groupwise_resolution_kwargs,

                post_registration_do_quality_filter=True,
                post_registration_quality_threshold=0.1,

                plot_summary=True,
                return_metrics=True
            )
            register_sims = [msi_utils.get_sim_from_msim(msim) for msim in register_msims]
            sims = sims0

            final_residual = list(metrics['mean_residual'])[-1]

            mappings_dict = {index: mapping.data[0] for index, mapping in zip(indices, mappings)}
            distances = [np.linalg.norm(param_utils.translation_from_affine(mapping)).item()
                         for mapping in mappings_dict.values()]

            if len(sims) > 2:
                # Coefficient of variation
                cvar = np.std(distances) / np.mean(distances)
                confidence = 1 - min(cvar / 10, 1)
            else:
                sim0 = sims[0]
                spatial_dims = si_utils.get_spatial_dims_from_sim(sim0)
                size = [sim0.sizes[dim] * si_utils.get_spacing_from_sim(sim0)[dim] for dim in spatial_dims]
                norm_distance = np.sum(distances) / np.linalg.norm(size)
                confidence = 1 - min(math.sqrt(norm_distance), 1)

            # copy transforms from register msims to unmodified msims
            for reg_sim, index in zip(register_sims, indices):
                si_utils.set_sim_affine(
                    sims[index],
                    si_utils.get_affine_from_sim(reg_sim, transform_key=reg_transform_key),
                    transform_key=reg_transform_key)

        except NotEnoughOverlapError:
            final_residual = 0
            confidence = 0
            for sim in sims:
                si_utils.set_sim_affine(
                    sim,
                    param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                    transform_key=reg_transform_key,
                    base_transform_key=source_transform_key)
            mappings = [param_utils.identity_transform(ndim=ndims, t_coords=[0])] * len(sims)
            mappings_dict = {index: np.eye(ndims + 1) for index, _ in enumerate(sims)}

    if verbose:
        progress.update()
        progress.close()

    logging.info('Fusing...')
    # convert to multichannel images
    if is_stack:
        # set 3D affine transforms from 2D registration params
        for index, sim in enumerate(sims):
            affine_3d = param_utils.identity_transform(ndim=3)
            affine_3d.loc[{dim: mappings[index].coords[dim] for dim in mappings[index].sel(t=0).dims}] = mappings[index].sel(t=0)
            si_utils.set_sim_affine(sim, affine_3d, transform_key=reg_transform_key, base_transform_key=source_transform_key)

        output_spacing = si_utils.get_spacing_from_sim(sims[0]) | {'z': z_scale}
        # calculate output stack properties from input views
        output_stack_properties = fusion.calc_stack_properties_from_view_properties_and_params(
            [si_utils.get_stack_properties_from_sim(sim) for sim in sims],
            [np.array(si_utils.get_affine_from_sim(sim, reg_transform_key).squeeze()) for sim in sims],
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
            transform_key=reg_transform_key,
            output_stack_properties=output_stack_properties,
            fusion_func=fusion.simple_average_fusion,
        )
    elif is_channel_overlay:
        output_stack_properties = si_utils.get_stack_properties_from_sim(sims[0])
        channel_sims = [fusion.fuse(
            [sim],
            transform_key=reg_transform_key,
            output_stack_properties=output_stack_properties
        ) for sim in sims]
        channel_sims = [sim.assign_coords({'c': [channels[simi]['label']]}) for simi, sim in enumerate(channel_sims)]
        fused_image = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
    else:
        fused_image = fusion.fuse(
            sims,
            transform_key=reg_transform_key
        )
    return {'mappings': mappings_dict,
            'final_residual': final_residual,
            'confidence': confidence,
            'sims': sims,
            'fused_image': fused_image}


def run_operation(params, params_general):
    operation = params['operation']
    filenames = dir_regex(params['input'])
    if len(filenames) == 0:
        logging.warning(f'Skipping operation {operation} (no files)')
        return

    operation_parts = operation.split()
    if 'match' in operation_parts:
        # sort last key first
        filenames = sorted(filenames, key=lambda file: list(reversed(find_all_numbers(get_filetitle(file)))))
        if len(operation_parts) > operation_parts.index('match') + 1:
            match_label = operation_parts[-1]
        else:
            match_label = 's'
        matches = {}
        for filename in filenames:
            parts = split_underscore_numeric(filename)
            match_value = parts.get(match_label)
            if match_value is not None:
                if match_value not in matches:
                    matches[match_value] = []
                matches[match_value].append(filename)
            if len(matches) == 0:
                matches[0] = filenames
        filesets = list(matches.values())
        fileset_labels = [match_label + label for label in matches.keys()]
    else:
        filesets = [filenames]
        fileset_labels = ['']
    for fileset, fileset_label in zip(filesets, fileset_labels):
        if len(filesets) > 1:
            logging.info(f'File set: {fileset_label}')
        run_operation_files(fileset, params, params_general)


def run_operation_files(filenames, params, params_general):
    operation = params['operation']
    is_stack = ('stack' in operation)
    is_transition = ('transition' in operation)
    output_params = params_general.get('output', {})
    invert_x_coordinates = params.get('invert_x_coordinates', False)
    normalise_orientation = params.get('normalise_orientation', False)
    reset_coordinates = params.get('reset_coordinates', False)
    extra_metadata = params.get('extra_metadata', {})
    channels = extra_metadata.get('channels', [])
    clear = output_params.get('clear', False)

    show_original = params_general.get('show_original', False)
    verbose = params_general.get('verbose', False)

    source_transform_key = 'stage_metadata'
    reg_transform_key = 'registered'
    transition_transform_key = 'transition'

    file_indices = ['-'.join(map(str, find_all_numbers(get_filetitle(filename))[-2:])) for filename in filenames]

    if len(filenames) == 0:
        logging.warning('Skipping (no tiles)')
        return

    input_dir = os.path.dirname(filenames[0])
    parts = split_underscore_numeric(filenames[0])
    output_pattern = params['output'].format_map(parts)
    output = os.path.join(input_dir, output_pattern)
    output_dir = os.path.dirname(output)
    if clear:
        shutil.rmtree(output_dir, ignore_errors=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info('Initialising tiles...')
    sims, positions, rotations = init_tiles(filenames, source_transform_key, d3=is_stack,
                                            invert_x_coordinates=invert_x_coordinates,
                                            normalise_orientation=normalise_orientation,
                                            reset_coordinates=reset_coordinates,
                                            verbose=verbose)

    registered_fused_filename = output + 'registered'
    if len(filenames) == 1:
        logging.warning('Skipping registration (single tile)')
        save_image(registered_fused_filename, sims[0], channels=channels, translation0=positions[0],
                   params=output_params)
        return

    if show_original:
        # before registration:
        logging.info('Plotting tiles...')
        msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]
        original_positions_filename = output + 'positions_original.png'
        vis_utils.plot_positions(msims, transform_key=source_transform_key, use_positional_colors=False,
                                 view_labels=file_indices, view_labels_size=3,
                                 show_plot=verbose, output_filename=original_positions_filename)

        logging.info('Fusing original...')
        original_fused = fusion.fuse(sims, transform_key=source_transform_key)
        original_fused_filename = output + 'original'
        save_image(original_fused_filename, original_fused, transform_key=source_transform_key,
                   params=output_params)

    results = register(sims, source_transform_key, reg_transform_key, params, params_general)
    fused_image = results['fused_image']
    sims = results['sims']
    mappings = results['mappings']
    mappings2 = {get_filetitle(filenames[index]): mapping.tolist() for index, mapping in mappings.items()}
    metrics = f'Final residual: {results["final_residual"]:.3f} Confidence: {results["confidence"]:.3f}'
    logging.info(metrics)

    with open(output + 'mappings.json', 'w') as file:
        json.dump(mappings2, file, indent=4)

    with open(output + 'mappings.csv', 'w', newline='') as file:
        csvwriter = csv.writer(file)
        header = ['Tile', 'x', 'y', 'z', 'rotation']
        csvwriter.writerow(header)
        if verbose:
            print(header)
        for sim, (index, mapping), position, rotation in zip(sims, mappings.items(), positions, rotations):
            if not normalise_orientation:
                # rotation already in msim affine transform
                rotation = None
            position, rotation = get_data_mapping(sim, transform_key=reg_transform_key,
                                                  transform=mapping, translation0=position, rotation=rotation)
            row = [get_filetitle(filenames[index])] + list(position) + [rotation]
            csvwriter.writerow(row)
            if verbose:
                print(row)

    with open(output + 'metrics.txt', 'w') as file:
        file.write(metrics)

    if verbose:
        print('Mappings:')
        print(mappings2)

    logging.info('Saving fused image...')
    save_image(registered_fused_filename, fused_image,
               transform_key=reg_transform_key, channels=channels, translation0=positions[0],
               params=output_params)

    # plot the tile configuration after registration
    logging.info('Plotting tiles...')
    registered_positions_filename = output + 'positions_registered.png'
    vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim) for sim in sims], transform_key=reg_transform_key, use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=verbose, output_filename=registered_positions_filename)

    if is_transition:
        logging.info('Creating transition...')
        pixel_size = pixel_size_xyz[:2]
        nframes = params.get('frames', 1)
        spacing = params.get('spacing', [1.1, 1])
        scale = params.get('scale', 1)
        transition_filename = output + 'transition'
        video = Video(transition_filename + '.mp4', fps=params.get('fps', 1))
        positions0 = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in sims])
        center = np.mean(positions0, 0)
        window = get_image_window(fused_image)

        max_size = None
        acum = 0
        for framei in range(nframes):
            c = (1 - np.cos(framei / (nframes - 1) * 2 * math.pi)) / 2
            acum += c / (nframes / 2)
            spacing1 = spacing[0] + (spacing[1] - spacing[0]) * acum
            for sim, position0 in zip(sims, positions0):
                transform = param_utils.identity_transform(ndim=2, t_coords=[0])
                transform[0][:2, 2] += (position0 - center) * spacing1
                si_utils.set_sim_affine(sim, transform, transform_key=transition_transform_key)
            frame = fusion.fuse(sims, transform_key=transition_transform_key).squeeze()
            frame = float2int_image(normalise_values(frame, window[0], window[1]))
            frame = cv.resize(np.asarray(frame), None, fx=scale, fy=scale)
            if max_size is None:
                max_size = frame.shape[1], frame.shape[0]
                video.size = max_size
            frame = image_reshape(frame, max_size)
            save_tiff(transition_filename + f'{framei:04d}.tiff', frame, None, pixel_size)
            video.write(frame)

        video.close()


def run(params):
    logging.info(f'Multiview-stitcher Version: {multiview_stitcher.__version__}')

    params_general = params['general']
    break_on_error = params_general.get('break_on_error', False)

    for operation in tqdm(params['operations']):
        input_path = operation['input']
        logging.info(f'Input: {input_path}')
        try:
            run_operation(operation, params_general)
        except Exception as e:
            logging.exception(f'Error processing: {input_path}')
            print(f'Error processing: {input_path}: {e}')
            if break_on_error:
                break

    logging.info('Done!')
