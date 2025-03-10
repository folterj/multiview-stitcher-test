# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

import csv
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
import shutil
from tqdm import tqdm
import xarray as xr

from src.Video import Video
from src.image.flatfield import flatfield_correction
from src.image.ome_helper import save_image
from src.image.ome_tiff_helper import save_tiff
from src.image.source_helper import create_source
from src.image.util import *
from src.util import *


class MVSRegistration:
    def __init__(self, params_general):
        self.params_general = params_general
        self.verbose = self.params_general.get('verbose', False)
        self.verbose_mvs = self.params_general.get('verbose_mvs', False)

        self.source_transform_key = 'stage_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'

        logging.info(f'Multiview-stitcher Version: {multiview_stitcher.__version__}')

    def run_operation(self, filenames, params):
        operation = params['operation']
        is_stack = ('stack' in operation)
        is_transition = ('transition' in operation)
        normalise_orientation = params.get('normalise_orientation', False)
        extra_metadata = params.get('extra_metadata', {})
        channels = extra_metadata.get('channels', [])

        show_original = self.params_general.get('show_original', False)
        output_params = self.params_general.get('output', {})
        clear = output_params.get('clear', False)
        overwrite = output_params.get('overwrite', False)

        file_labels = []
        for filename in filenames:
            file_label = '-'.join(map(str, find_all_numbers(filename)[-2:]))
            if file_label == '':
                file_label = get_filetitle(filename)
            file_labels.append(file_label)

        if len(filenames) == 0:
            logging.warning('Skipping (no tiles)')
            return

        input_dir = os.path.dirname(filenames[0])
        parts = split_underscore_numeric(filenames[0])
        output_pattern = params['output'].format_map(parts)
        output = os.path.join(input_dir, output_pattern)    # preserve trailing slash: do not use os.path.normpath()
        output_dir = os.path.dirname(output)
        if not overwrite and (os.path.exists(output_dir) and os.listdir(output_dir)):
            logging.warning(f'Non-empty output directory {os.path.normpath(output_dir)} skipped')
            return
        if clear:
            shutil.rmtree(output_dir, ignore_errors=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sims, positions, rotations = self.init_tiles(filenames, params)

        registered_fused_filename = output + 'registered'
        if len(filenames) == 1:
            logging.warning('Skipping registration (single tile)')
            save_image(registered_fused_filename, sims[0], channels=channels, translation0=positions[0],
                       params=output_params)
            return

        if show_original:
            # before registration:
            logging.info('Plotting tiles...')
            original_positions_filename = output + 'positions_original.pdf'
            vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim) for sim in sims], transform_key=self.source_transform_key,
                                     use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                     show_plot=self.verbose, output_filename=original_positions_filename)

            logging.info('Fusing original...')
            if is_stack:
                sims2d = [si_utils.max_project_sim(sim, dim='z') for sim in sims]
            else:
                sims2d = sims
            original_fused = fusion.fuse(sims2d, transform_key=self.source_transform_key)
            original_fused_filename = output + 'original'
            save_image(original_fused_filename, original_fused, transform_key=self.source_transform_key,
                       params=output_params)

        results = self.register(sims, params)

        fused_image = results['fused_image']
        sims = results['sims']
        mappings = results['mappings']
        mappings2 = {file_labels[index]: mapping.tolist() for index, mapping in mappings.items()}

        reg_result = results['reg_result']
        pairwise_registration_results = reg_result.get('pairwise_registration', {})
        groupwise_resolution_results = reg_result.get('groupwise_resolution', {})
        registration_quality = np.mean(list(pairwise_registration_results.get('metrics', {}).get('qualities', {}).values()))
        residual_error = np.mean(list(groupwise_resolution_results.get('metrics', {}).get('residuals', {}).values()))
        metrics = (f'Residual error: {residual_error:.3f}'
                   f' Registration quality: {registration_quality:.3f}'
                   f' Confidence: {results["confidence"]:.3f}')
        logging.info(metrics)

        with open(output + 'mappings.json', 'w') as file:
            json.dump(mappings2, file, indent=4)

        if self.verbose:
            print('Mappings:')
            print(mappings2)

        with open(output + 'mappings.csv', 'w', newline='') as file:
            csvwriter = csv.writer(file)
            header = ['Tile', 'x', 'y', 'z', 'rotation']
            csvwriter.writerow(header)
            if self.verbose:
                print(header)
            for sim, (index, mapping), position, rotation in zip(sims, mappings.items(), positions, rotations):
                if not normalise_orientation:
                    # rotation already in msim affine transform
                    rotation = None
                position, rotation = get_data_mapping(sim, transform_key=self.reg_transform_key,
                                                      transform=mapping, translation0=position, rotation=rotation)
                row = [file_labels[index]] + list(position) + [rotation]
                csvwriter.writerow(row)
                if self.verbose:
                    print(row)

        with open(output + 'metrics.txt', 'w') as file:
            file.write(metrics)

        logging.info('Saving fused image...')
        save_image(registered_fused_filename, fused_image,
                   transform_key=self.reg_transform_key, channels=channels, translation0=positions[0],
                   params=output_params)

        # plot the tile configuration after registration
        logging.info('Plotting tiles...')
        registered_positions_filename = output + 'positions_registered.pdf'
        vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim) for sim in sims], transform_key=self.reg_transform_key,
                                 use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                 show_plot=self.verbose, output_filename=registered_positions_filename)

        summary_plot = pairwise_registration_results.get('summary_plot')
        if summary_plot is not None:
            figure, axes = summary_plot
            summary_plot_filename = output + 'pairwise_registration.pdf'
            figure.savefig(summary_plot_filename)

        summary_plot = groupwise_resolution_results.get('summary_plot')
        if summary_plot is not None:
            figure, axes = summary_plot
            summary_plot_filename = output + 'groupwise_resolution.pdf'
            figure.savefig(summary_plot_filename)

        if is_transition:
            logging.info('Creating transition...')
            pixel_size = [si_utils.get_spacing_from_sim(sims[0]).get(dim, 1) for dim in 'xy']
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
                    si_utils.set_sim_affine(sim, transform, transform_key=self.transition_transform_key)
                frame = fusion.fuse(sims, transform_key=self.transition_transform_key).squeeze()
                frame = float2int_image(normalise_values(frame, window[0], window[1]))
                frame = cv.resize(np.asarray(frame), None, fx=scale, fy=scale)
                if max_size is None:
                    max_size = frame.shape[1], frame.shape[0]
                    video.size = max_size
                frame = image_reshape(frame, max_size)
                save_tiff(transition_filename + f'{framei:04d}.tiff', frame, None, pixel_size)
                video.write(frame)

            video.close()

    def init_tiles(self, filenames, params):
        operation = params['operation']
        is_stack = ('stack' in operation)
        invert_x_coordinates = params.get('invert_x_coordinates', False)
        normalise_orientation = params.get('normalise_orientation', False)
        reset_coordinates = params.get('reset_coordinates', False)
        extra_metadata = params.get('extra_metadata', {})
        z_scale = extra_metadata.get('scale', {}).get('z', 1)

        sources = [create_source(file) for file in filenames]
        images = []
        sims = []
        translations = []
        rotations = []

        source0 = sources[0]
        output_order = 'zyx' if is_stack else 'yx'
        if source0.get_nchannels() > 1:
            output_order += 'c'

        last_z_position = None
        different_z_positions = False
        for source in tqdm(sources, disable=not self.verbose, desc='Initialising tiles'):
            if reset_coordinates or len(source.get_position()) == 0:
                translation = np.zeros(3)
            else:
                translation = np.array(source.get_position_micrometer())
                if invert_x_coordinates:
                    translation[0] = -translation[0]
                    translation[1] = -translation[1]
            if len(translation) >= 3:
                z_position = translation[2]
            else:
                z_position = 0
            if last_z_position is not None and z_position != last_z_position:
                different_z_positions = True
            translations.append(translation)
            rotations.append(source.get_rotation())
            image = redimension_data(source.get_source_dask()[0],
                                     source.dimension_order, output_order)
            images.append(image)
            last_z_position = z_position

        increase_z_positions = is_stack and not different_z_positions

        if normalise_orientation:
            size = np.array(source0.get_size()) * source0.get_pixel_size_micrometer()
            translations, rotations = normalise_rotated_positions(translations, rotations, size)

        #translations = [np.array(translation) * 1.25 for translation in translations]

        z_position = 0
        for source, image, translation, rotation in zip(sources, images, translations, rotations):
            # transform #dimensions need to match
            scale_dict = convert_xyz_to_dict(source.get_pixel_size_micrometer())
            if len(scale_dict) > 0 and 'z' not in scale_dict:
                scale_dict['z'] = z_scale
            translation_dict = convert_xyz_to_dict(translation)
            if (len(translation_dict) > 0 and 'z' not in translation_dict) or increase_z_positions:
                translation_dict['z'] = z_position
                print('z_position', z_position)
            if increase_z_positions:
                z_position += z_scale
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
                transform_key=self.source_transform_key,
                c_coords=channel_labels
            )
            sims.append(sim.chunk({'y': 1024, 'x': 1024}))
        return sims, translations, rotations

    def register(self, sims, params):
        sim0 = sims[0]
        ndims = si_utils.get_ndim_from_sim(sim0)
        source_type = sim0.dtype
        size = si_utils.get_shape_from_sim(sim0, asarray=True) * si_utils.get_spacing_from_sim(sim0, asarray=True)

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
        show_filtered = self.params_general.get('show_filtered', False)

        extra_metadata = params.get('extra_metadata', {})
        channels = extra_metadata.get('channels', [])
        z_scale = extra_metadata.get('scale', {}).get('z', 1)
        is_stack = ('stack' in operation)
        is_channel_overlay = (len(channels) > 1)

        if flatfield_quantile is not None or filter_foreground:
            foreground_map = calc_foreground_map(sims)
        if flatfield_quantile is not None:
            sims = flatfield_correction(sims, self.source_transform_key, foreground_map, flatfield_quantile)

        # copy source to reg transform: in case not set by registration
        for sim in sims:
            si_utils.set_sim_affine(
                sim,
                param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                transform_key=self.reg_transform_key,
                base_transform_key=self.source_transform_key)

        if normalisation:
            use_global = not is_channel_overlay
            if use_global:
                logging.info('Normalising tiles (global)...')
            else:
                logging.info('Normalising tiles...')
            register_sims = normalise(sims, self.source_transform_key, use_global=use_global)
        else:
            register_sims = sims

        if filter_foreground:
            logging.info('Filtering foreground tiles...')
            #tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
            #threshold1 = np.mean(tile_vars)
            #threshold2 = np.median(tile_vars)
            #threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
            #threshold = min(threshold1, threshold2, threshold3)
            #foregrounds = (tile_vars >= threshold)
            register_sims = [sim for sim, is_foreground in zip(register_sims, foreground_map) if is_foreground]

            if show_filtered:
                filtered_filename = params['output'] + 'filtered.pdf'
                vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim) for sim in register_sims],
                                         transform_key=self.source_transform_key, use_positional_colors=False,
                                         view_labels_size=3, show_plot=self.verbose, output_filename=filtered_filename)

            logging.info(f'Foreground tiles: {len(register_sims)} / {len(sims)}')

            indices = np.where(foreground_map)[0]
        else:
            indices = range(len(sims))

        if is_stack:
            # register in 2d; pairwise consecutive views
            register_sims = [si_utils.max_project_sim(sim, dim='z') for sim in register_sims]
            pairs = [(index, index + 1) for index in range(len(register_sims) - 1)]
        elif use_orthogonal_pairs:
            origins = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in register_sims])
            pairs, _ = get_orthogonal_pairs_from_tiles(origins, size)
            logging.info(f'#pairs: {len(pairs)}')
        else:
            pairs = None

        if 'dummy' in method:
            from src.RegistrationMethodDummy import RegistrationMethodDummy
            registration_method = RegistrationMethodDummy(source_type)
            pairwise_reg_func = registration_method.registration
        elif 'feature' in method:
            from src.RegistrationMethodFeatures import RegistrationMethodFeatures
            registration_method = RegistrationMethodFeatures(source_type)
            pairwise_reg_func = registration_method.registration
        elif 'cpd' in method:
            from src.RegistrationMethodCPD import RegistrationMethodCPD
            registration_method = RegistrationMethodCPD(source_type)
            pairwise_reg_func = registration_method.registration
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

        if self.verbose:
            progress = tqdm(desc='Registering', total=1)

        with ProgressBar() if self.verbose_mvs else nullcontext():
            try:
                logging.info('Registering...')
                reg_result = registration.register(
                    register_msims,
                    reg_channel=reg_channel,
                    reg_channel_index=reg_channel_index,
                    transform_key=self.source_transform_key,
                    new_transform_key=self.reg_transform_key,

                    pairs=pairs,
                    pre_registration_pruning_method=None,

                    pairwise_reg_func=pairwise_reg_func,
                    pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                    groupwise_resolution_kwargs=groupwise_resolution_kwargs,

                    post_registration_do_quality_filter=True,
                    post_registration_quality_threshold=0.1,

                    plot_summary=True,
                    return_dict=True
                )
                # copy transforms from register sims to unmodified sims
                for reg_msim, index in zip(register_msims, indices):
                    si_utils.set_sim_affine(
                        sims[index],
                        msi_utils.get_transform_from_msim(reg_msim, transform_key=self.reg_transform_key),
                        transform_key=self.reg_transform_key)

                mappings = reg_result['params']
                mappings_dict = {index: mapping.data[0] for index, mapping in zip(indices, mappings)}
                distances = [np.linalg.norm(param_utils.translation_from_affine(mapping)).item()
                             for mapping in mappings_dict.values()]

                if len(register_sims) > 2:
                    # Coefficient of variation
                    cvar = np.std(distances) / np.mean(distances)
                    confidence = 1 - min(cvar / 10, 1)
                else:
                    norm_distance = np.sum(distances) / np.linalg.norm(size)
                    confidence = 1 - min(math.sqrt(norm_distance), 1)
            except NotEnoughOverlapError:
                logging.warning('Not enough overlap')
                reg_result = {}
                mappings = [param_utils.identity_transform(ndim=ndims, t_coords=[0])] * len(sims)
                mappings_dict = {index: np.eye(ndims + 1) for index, _ in enumerate(sims)}
                confidence = 0

        if self.verbose:
            progress.update()
            progress.close()

        output_chunksize = {'y': 1024, 'x': 1024}
        for dim in sim0.dims:
            if dim not in output_chunksize:
                output_chunksize[dim] = 1

        if is_stack:
            # set 3D affine transforms from 2D registration params
            for index, sim in enumerate(sims):
                affine_3d = param_utils.identity_transform(ndim=3)
                affine_3d.loc[{dim: mappings[index].coords[dim] for dim in mappings[index].sel(t=0).dims}] = mappings[index].sel(t=0)
                si_utils.set_sim_affine(sim, affine_3d, transform_key=self.reg_transform_key, base_transform_key=self.source_transform_key)

            output_stack_properties = calc_output_properties(sims, self.reg_transform_key, z_scale=z_scale)
            # set z shape which is wrongly calculated by calc_stack_properties_from_view_properties_and_params
            # because it does not take into account the correct input z spacing because of stacks of one z plane
            if self.verbose:
                logging.info(f'Output stack: {output_stack_properties}')
            output_stack_properties['shape']['z'] = len(sims)
            # fuse all sims together using simple average fusion

            data_size = np.prod(list(output_stack_properties['shape'].values())) * source_type.itemsize
            logging.info(f'Fusing Z stack {print_hbytes(data_size)}')

            fused_image = fusion.fuse(
                sims,
                transform_key=self.reg_transform_key,
                output_stack_properties=output_stack_properties,
                output_chunksize=output_chunksize,
                fusion_func=fusion.simple_average_fusion,
            )
        elif is_channel_overlay:
            # convert to multichannel images
            output_stack_properties = calc_output_properties(sims, self.reg_transform_key)
            data_size = np.prod(list(output_stack_properties['shape'].values())) * len(sims) * source_type.itemsize
            logging.info(f'Fusing channels {print_hbytes(data_size)}')

            channel_sims = [fusion.fuse(
                [sim],
                transform_key=self.reg_transform_key,
                output_chunksize=output_chunksize,
                output_stack_properties=output_stack_properties
            ) for sim in sims]
            channel_sims = [sim.assign_coords({'c': [channels[simi]['label']]}) for simi, sim in enumerate(channel_sims)]
            fused_image = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
        else:
            output_stack_properties = calc_output_properties(sims, self.reg_transform_key)
            data_size = np.prod(list(output_stack_properties['shape'].values())) * source_type.itemsize
            logging.info(f'Fusing {print_hbytes(data_size)}')

            fused_image = fusion.fuse(
                sims,
                transform_key=self.reg_transform_key,
                output_chunksize=output_chunksize,
            )
        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'confidence': confidence,
                'sims': sims,
                'fused_image': fused_image}
