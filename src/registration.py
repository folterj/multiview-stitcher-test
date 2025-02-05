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
from src.image.ome_helper import save_image
from src.image.util import *
from src.util import *


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


def init_tiles(files, flatfield_quantile=None,
               invert_x_coordinates=False, normalise_orientation=False, reset_coordinates=False,
               verbose=False):
    sims = []
    sources = [create_source(file) for file in files]
    nchannels = sources[0].get_nchannels()
    images = []
    logging.info('Init tiles...')
    for source in tqdm(sources, disable=not verbose):
        output_order = 'zyx'
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
            transform_key="stage_metadata",
            c_coords=channel_labels
        )
        sims.append(sim)
        source.close()

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
            affine=si_utils.get_affine_from_sim(sim, 'stage_metadata'),
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


def register(sims0, operation, method, reg_channel=None, reg_channel_index=None, normalisation=False, filter_foreground=False,
             use_orthogonal_pairs=False, use_rotation=False, extra_metadata={}, verbose=False):
    if isinstance(reg_channel, int):
        reg_channel_index = reg_channel
        reg_channel = None

    global pixel_size_xyz
    sim0 = sims0[0]
    pixel_size_xyz = [si_utils.get_spacing_from_sim(sim0).get(dim, 1) for dim in 'xyz']

    channels = extra_metadata.get('channels', [])
    z_scale = extra_metadata.get('scale', {}).get('z', 1)
    is_stack = ('stack' in operation)
    is_channel_overlay = (len(channels) > 1)
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

    if is_stack:
        pairs = [(index, index + 1) for index in range(len(register_msims) - 1)]
    elif use_orthogonal_pairs:
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
                    # options include 'Translation', 'Rigid', 'Affine', 'Similarity' and can be concatenated
                    #"aff_metric": "meansquares",  # options include 'mattes', 'meansquares',
                    # more parameters to tune:
                    "aff_random_sampling_rate": 0.5,
                    "aff_iterations": (2000, 2000, 1000, 1000),
                    "aff_smoothing_sigmas": (4, 2, 1, 0),
                    "aff_shrink_factors": (16, 8, 2, 1),

                    # "aff_random_sampling_rate": 0.2,
                    # "aff_iterations": (2000, 2000),
                    # "aff_smoothing_sigmas": (1, 0),
                    # "aff_shrink_factors": (2, 1),
                }
                # these are the parameters for the groupwise registration (global optimization)
                groupwise_resolution_kwargs = {
                    'transform': 'rigid',  # options include 'translation', 'rigid', 'affine'
                }
            else:
                pairwise_reg_func_kwargs = None
                groupwise_resolution_kwargs = None

            mappings, metrics = registration.register(
                register_msims,
                reg_channel=reg_channel,
                reg_channel_index=reg_channel_index,
                transform_key="stage_metadata",
                new_transform_key="registered",

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

            final_residual = list(metrics['mean_residual'])[-1]

            mappings_dict = {index: mapping.data[0] for index, mapping in zip(indices, mappings)}
            distances = [np.linalg.norm(param_utils.translation_from_affine(mapping)).item()
                         for mapping in mappings_dict.values()]

            if len(sims0) > 2:
                # Coefficient of variation
                cvar = np.std(distances) / np.mean(distances)
                confidence = 1 - min(cvar / 10, 1)
            else:
                sim0 = sims0[0]
                spatial_dims = si_utils.get_spatial_dims_from_sim(sim0)
                size = [sim0.sizes[dim] * si_utils.get_spacing_from_sim(sim0)[dim] for dim in spatial_dims]
                norm_distance = np.sum(distances) / np.linalg.norm(size)
                confidence = 1 - min(math.sqrt(norm_distance), 1)

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
            mappings_dict = {index: np.eye(3) for index, _ in enumerate(msims0)}

    if verbose:
        progress.update()
        progress.close()

    logging.info('Fusing...')
    # convert to multichannel images
    sims0 = [msi_utils.get_sim_from_msim(msim) for msim in msims0]
    if is_stack:
        output_stack_properties = si_utils.get_stack_properties_from_sim(sims0[0])
        stack_sims = [fusion.fuse(
            [sim],
            transform_key='registered',
            output_stack_properties=output_stack_properties
        ) for sim in sims0]
        #stack_sims = [sim.assign_coords({'z': simi * z_scale}) for simi, sim in enumerate(stack_sims)]
        stack_sims = [sim.expand_dims(axis=2, z=[simi * z_scale]) for simi, sim in enumerate(stack_sims)]
        fused_image = xr.combine_nested([sim.rename() for sim in stack_sims], concat_dim='z', combine_attrs='override')
    elif is_channel_overlay:
        output_stack_properties = si_utils.get_stack_properties_from_sim(sims0[0])
        channel_sims = [fusion.fuse(
            [sim],
            transform_key='registered',
            output_stack_properties=output_stack_properties
        ) for sim in sims0]
        channel_sims = [sim.assign_coords({'c': [channels[simi]['label']]}) for simi, sim in enumerate(channel_sims)]
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
    output_params = params_general.get('output', {})
    method = params.get('method', '').lower()
    invert_x_coordinates = params.get('invert_x_coordinates', False)
    flatfield_quantile = params.get('flatfield_quantile')
    normalisation = params.get('normalisation', False)
    filter_foreground = params.get('filter_foreground', False)
    use_orthogonal_pairs = params.get('use_orthogonal_pairs', False)
    normalise_orientation = params.get('normalise_orientation', False)
    reset_coordinates = params.get('reset_coordinates', False)
    use_rotation = params.get('use_rotation', False)
    reg_channel = params.get('channel', 0)
    extra_metadata = params.get('extra_metadata', {})
    channels = extra_metadata.get('channels', [])
    clear = output_params.get('clear', False)

    show_original = params_general.get('show_original', False)
    npyramid_add = params_general.get('npyramid_add', 0)
    pyramid_downsample = params_general.get('pyramid_downsample', 2)
    verbose = params_general.get('verbose', False)

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

    original_positions_filename = output + 'positions_original.png'
    original_fused_filename = output + 'original'
    registered_positions_filename = output + 'positions_registered.png'
    registered_fused_filename = output + 'registered'

    logging.info('Initialising tiles...')
    sims, positions, rotations = init_tiles(filenames, flatfield_quantile=flatfield_quantile,
                                            invert_x_coordinates=invert_x_coordinates,
                                            normalise_orientation=normalise_orientation,
                                            reset_coordinates=reset_coordinates,
                                            verbose=verbose)

    if len(filenames) == 1:
        logging.warning('Skipping registration (single tile)')
        save_image(registered_fused_filename, sims[0], channels=channels, translation0=positions[0],
                   npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample, params=output_params)
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

    results = register(sims, operation, method, reg_channel, normalisation=normalisation, filter_foreground=filter_foreground,
                       use_orthogonal_pairs=use_orthogonal_pairs, use_rotation=use_rotation, extra_metadata=extra_metadata,
                       verbose=verbose)
    msims = results['msims']
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
        for msim, (index, mapping), position, rotation in zip(msims, mappings.items(), positions, rotations):
            if not normalise_orientation:
                # rotation already in msim affine transform
                rotation = None
            position, rotation = get_data_mapping(msim, transform_key='registered',
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

    # plot the tile configuration after registration
    logging.info('Plotting tiles...')
    vis_utils.plot_positions(msims, transform_key='registered', use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=False, output_filename=registered_positions_filename)

    logging.info('Saving fused image...')
    save_image(registered_fused_filename, results['fused_image'],
               transform_key='registered', channels=channels, translation0=positions[0],
               npyramid_add=npyramid_add, pyramid_downsample=pyramid_downsample, params=output_params)


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
