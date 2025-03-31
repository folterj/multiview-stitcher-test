# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

import csv
from contextlib import nullcontext
from dask.diagnostics import ProgressBar
import json
import logging
import multiview_stitcher
from multiview_stitcher import registration, vis_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
import shutil
from tqdm import tqdm
import xarray as xr

from src.Video import Video
from src.image.flatfield import flatfield_correction
from src.image.ome_helper import save_image, exists_output
from src.image.ome_tiff_helper import save_tiff
from src.image.source_helper import create_source
from src.image.util import *
from src.util import *


class MVSRegistration:
    def __init__(self, params_general):
        self.params_general = params_general
        self.verbose = self.params_general.get('verbose', False)
        self.verbose_mvs = self.params_general.get('verbose_mvs', False)

        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'

        logging.info(f'Multiview-stitcher version: {multiview_stitcher.__version__}')

    def run_operation(self, filenames, params, global_rotation=None, global_center=None):
        operation = params['operation']
        overlap_threshold = params.get('overlap_threshold', 0.5)
        source_metadata = params.get('source_metadata', {})
        extra_metadata = params.get('extra_metadata', {})
        channels = extra_metadata.get('channels', [])
        normalise_orientation = 'norm' in source_metadata

        show_original = self.params_general.get('show_original', False)
        output_params = self.params_general.get('output', {})
        clear = output_params.get('clear', False)
        overwrite = output_params.get('overwrite', True)

        is_stack = ('stack' in operation)
        is_transition = ('transition' in operation)
        is_channel_overlay = (len(channels) > 1)

        # get unique labels from filenames
        file_labels = []
        for filename in filenames:
            file_label = '-'.join(map(str, find_all_numbers(filename)[-2:]))
            if file_label == '':
                file_label = get_filetitle(filename)
            file_labels.append(file_label)
        if len(set(file_labels)) < len(file_labels):
            # duplicate labels: get unique differences
            file_labels = []
            last_filename = filenames[1]
            for filename in filenames:
                unique = set(split_path(filename)) - set(split_path(last_filename))
                file_labels.append('_'.join(unique))
                last_filename = filename

        if len(filenames) == 0:
            logging.warning('Skipping (no images)')
            return

        input_dir = os.path.dirname(filenames[0])
        parts = split_underscore_numeric(filenames[0])
        output_pattern = params['output'].format_map(parts)
        output = os.path.join(input_dir, output_pattern)    # preserve trailing slash: do not use os.path.normpath()
        registered_fused_filename = output + 'registered'

        output_dir = os.path.dirname(output)
        if not overwrite and exists_output(registered_fused_filename):
            logging.warning(f'Skipping existing output {os.path.normpath(output_dir)}')
            return
        if clear:
            shutil.rmtree(output_dir, ignore_errors=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sims, positions, rotations = self.init_sims(filenames, params,
                                                    global_rotation=global_rotation, global_center=global_center)

        with open(output + 'prereg_mappings.csv', 'w', newline='') as file:
            csvwriter = csv.writer(file)
            header = ['Image', 'x', 'y', 'z', 'rotation']
            csvwriter.writerow(header)
            if self.verbose:
                print('Pre-reg mappings')
                print('\t'.join(header))
            for label, sim in zip(file_labels, sims):
                position, rotation = get_data_mapping(sim, transform_key=self.source_transform_key)
                row = [label] + list(position) + [rotation]
                csvwriter.writerow(row)
                if self.verbose:
                    print('\t'.join(map(str, row)))

        if len(filenames) == 1:
            logging.warning('Skipping registration (single image)')
            save_image(registered_fused_filename, sims[0], channels=channels, translation0=positions[0],
                       params=output_params, verbose=self.verbose)
            return

        overlaps = self.validate_overlap(sims, file_labels, is_stack, is_stack or is_channel_overlay)
        overall_overlap = np.mean(overlaps)
        if overall_overlap < overlap_threshold:
            raise ValueError(f'Not enough overlap: {overall_overlap * 100:.1f}%')

        if show_original:
            # before registration:
            logging.info('Plotting images...')
            original_positions_filename = output + 'positions_original.pdf'

            if self.verbose:
                progress = tqdm(desc='Plotting original', total=1)
            vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim) for sim in sims], transform_key=self.source_transform_key,
                                     use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                     show_plot=self.verbose, output_filename=original_positions_filename)
            if self.verbose:
                progress.update()
                progress.close()

            logging.info('Fusing original...')
            if self.verbose:
                progress = tqdm(desc='Fusing original', total=1)
            if is_stack:
                sims2d = [si_utils.max_project_sim(sim, dim='z') for sim in sims]
            else:
                sims2d = sims
            original_fused = fusion.fuse(sims2d, transform_key=self.source_transform_key)
            if self.verbose:
                progress.update()
                progress.close()

            original_fused_filename = output + 'original'
            save_image(original_fused_filename, original_fused, transform_key=self.source_transform_key,
                       params=output_params)

        register_sims, indices = self.preprocess(sims, params)
        results = self.register(sims, register_sims, indices, params)

        reg_result = results['reg_result']
        sims = results['sims']
        mappings = results['mappings']
        mappings2 = {file_labels[index]: mapping.data[0].tolist() for index, mapping in mappings.items()}
        results['residual_errors'] = {file_labels[key[0]] + ' - ' + file_labels[key[1]]: value for key, value in results['residual_errors'].items()}
        results['registration_qualities'] = {file_labels[key[0]] + ' - ' + file_labels[key[1]]: value for key, value in results['registration_qualities'].items()}

        metrics = self.calc_metrics(results)

        logging.info(metrics['summary'])
        with open(output + 'metrics.json', 'w') as file:
            json.dump(metrics, file, indent=4)

        with open(output + 'mappings.json', 'w') as file:
            json.dump(mappings2, file, indent=4)

        if self.verbose:
            print('Mappings:')
            for key, value in mappings2.items():
                print(f'{key}: {value}')

        with open(output + 'mappings.csv', 'w', newline='') as file:
            csvwriter = csv.writer(file)
            header = ['Image', 'x', 'y', 'z', 'rotation']
            csvwriter.writerow(header)
            if self.verbose:
                print('\t'.join(header))
            for sim, (label, mapping), position, rotation in zip(sims, mappings2.items(), positions, rotations):
                if not normalise_orientation:
                    # rotation already in msim affine transform
                    rotation = None
                position, rotation = get_data_mapping(sim, transform_key=self.reg_transform_key,
                                                      transform=np.array(mapping),
                                                      translation0=position,
                                                      rotation=rotation)
                row = [label] + list(position) + [rotation]
                csvwriter.writerow(row)
                if self.verbose:
                    print('\t'.join(map(str, row)))

        # plot the image configuration after registration
        logging.info('Plotting images...')
        registered_positions_filename = output + 'positions_registered.pdf'
        if self.verbose:
            progress = tqdm(desc='Plotting registered', total=1)
        vis_utils.plot_positions([msi_utils.get_msim_from_sim(sim) for sim in sims], transform_key=self.reg_transform_key,
                                 use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                 show_plot=self.verbose, output_filename=registered_positions_filename)
        if self.verbose:
            progress.update()
            progress.close()

        summary_plot = reg_result.get('pairwise_registration', {}).get('summary_plot')
        if summary_plot is not None:
            figure, axes = summary_plot
            summary_plot_filename = output + 'pairwise_registration.pdf'
            figure.savefig(summary_plot_filename)

        summary_plot = reg_result.get('groupwise_resolution', {}).get('summary_plot')
        if summary_plot is not None:
            figure, axes = summary_plot
            summary_plot_filename = output + 'groupwise_resolution.pdf'
            figure.savefig(summary_plot_filename)

        fused_image = self.fuse(sims, params)
        logging.info('Saving fused image...')
        save_image(registered_fused_filename, fused_image,
                   transform_key=self.reg_transform_key, channels=channels, translation0=positions[0],
                   params=output_params, verbose=self.verbose)

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

    def init_sims(self, filenames, params, global_rotation=None, global_center=None):
        operation = params['operation']
        source_metadata = params.get('source_metadata', 'source')
        extra_metadata = params.get('extra_metadata', {})
        z_scale = extra_metadata.get('scale', {}).get('z')

        sources = [create_source(file) for file in filenames]
        source0 = sources[0]
        images = []
        sims = []
        translations = []
        rotations = []

        is_stack = ('stack' in operation)
        is_3d = (source0.get_size_xyzct()[2] > 1)

        output_order = 'zyx' if is_stack or is_3d else 'yx'
        ndims = len(output_order)
        if source0.get_nchannels() > 1:
            output_order += 'c'

        last_z_position = None
        different_z_positions = False
        delta_zs = []
        for source in tqdm(sources, disable=not self.verbose, desc='Initialising sims'):
            if isinstance(source_metadata, dict):
                translation = [source_metadata.get('x', 0), source_metadata.get('y', 0), source_metadata.get('z', 0)]
            else:
                translation = np.array(source.get_position_micrometer())
                if 'invert' in source_metadata:
                    translation[0] = -translation[0]
                    translation[1] = -translation[1]
            if len(translation) >= 3:
                z_position = translation[2]
            else:
                z_position = 0
            if last_z_position is not None and z_position != last_z_position:
                different_z_positions = True
                delta_zs.append(z_position - last_z_position)
            translations.append(translation)
            if global_rotation is not None:
                rotation = global_rotation
            else:
                rotation = source.get_rotation()
            rotations.append(rotation)
            image = redimension_data(source.get_source_dask()[0], source.dimension_order, output_order)
            images.append(image)
            last_z_position = z_position
        if z_scale is None:
            if len(delta_zs) > 0:
                z_scale = np.min(delta_zs)
            else:
                z_scale = 1

        if 'norm' in source_metadata:
            size = np.array(source0.get_size()) * source0.get_pixel_size_micrometer()
            center = None
            if 'center' in source_metadata:
                if 'global' in source_metadata:
                    center = global_center
                else:
                    center = np.mean(translations, 0)
            elif 'origin' in source_metadata:
                center = np.zeros(ndims)
            translations, rotations = normalise_rotated_positions(translations, rotations, size, center)

        #translations = [np.array(translation) * 1.25 for translation in translations]

        increase_z_positions = is_stack and not different_z_positions
        z_position = 0
        for source, image, translation, rotation in zip(sources, images, translations, rotations):
            # transform #dimensions need to match
            scale_dict = convert_xyz_to_dict(source.get_pixel_size_micrometer())
            if len(scale_dict) > 0 and 'z' not in scale_dict:
                scale_dict['z'] = abs(z_scale)
            translation_dict = convert_xyz_to_dict(translation)
            if (len(translation_dict) > 0 and 'z' not in translation_dict) or increase_z_positions:
                translation_dict['z'] = z_position
            if increase_z_positions:
                z_position += z_scale
            channel_labels = [channel.get('label', '') for channel in source.get_channels()]
            if rotation is None or 'norm' in source_metadata:
                transform = None
            else:
                transform = param_utils.invert_coordinate_order(
                    create_transform(translation, rotation, matrix_size=ndims + 1)
                )
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

    def validate_overlap(self, sims, labels, is_stack=False, expect_large_overlap=False):
        overlaps = []
        n = len(sims)
        positions = [si_utils.get_origin_from_sim(sim, asarray=True) for sim in sims]
        sizes = [np.linalg.norm(get_sim_physical_size(sim)) for sim in sims]
        if self.verbose:
            print('Overlap:')
        for i in range(n):
            norm_dists = []
            if is_stack:
                if i + 1 < n:
                    compare_indices = [i + 1]
                else:
                    compare_indices = []
            else:
                compare_indices = range(n)
            for j in compare_indices:
                if not j == i:
                    distance = math.dist(positions[i], positions[j])
                    norm_dist = distance / np.mean([sizes[i], sizes[j]])
                    norm_dists.append(norm_dist)
            if len(norm_dists) > 0:
                norm_dist = min(norm_dists)
                if self.verbose:
                    print(f'{labels[i]} norm distance: {norm_dist:.3f}')
                if norm_dist >= 1:
                    logging.warning(f'{labels[i]} has no overlap')
                    overlaps.append(False)
                elif expect_large_overlap and norm_dist > 0.5:
                    logging.warning(f'{labels[i]} has small overlap')
                    overlaps.append(False)
                else:
                    overlaps.append(True)
        return overlaps

    def preprocess(self, sims, params):
        flatfield_quantile = params.get('flatfield_quantile')
        normalisation = params.get('normalisation', False)
        filter_foreground = params.get('filter_foreground', False)
        extra_metadata = params.get('extra_metadata', {})
        channels = extra_metadata.get('channels', [])

        is_channel_overlay = (len(channels) > 1)

        if flatfield_quantile is not None or filter_foreground:
            foreground_map = calc_foreground_map(sims)
        if flatfield_quantile is not None:
            sims = flatfield_correction(sims, self.source_transform_key, foreground_map, flatfield_quantile)

        if normalisation:
            use_global = not is_channel_overlay
            if use_global:
                logging.info('Normalising image (global)...')
            else:
                logging.info('Normalising images...')
            register_sims = normalise(sims, self.source_transform_key, use_global=use_global)
        else:
            register_sims = sims

        if filter_foreground:
            logging.info('Filtering foreground images...')
            #tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
            #threshold1 = np.mean(tile_vars)
            #threshold2 = np.median(tile_vars)
            #threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
            #threshold = min(threshold1, threshold2, threshold3)
            #foregrounds = (tile_vars >= threshold)
            register_sims = [sim for sim, is_foreground in zip(register_sims, foreground_map) if is_foreground]
            logging.info(f'Foreground images: {len(register_sims)} / {len(sims)}')
            indices = np.where(foreground_map)[0]
        else:
            indices = range(len(sims))
        return register_sims, indices

    def register(self, sims, register_sims, indices, params):
        sim0 = sims[0]
        ndims = si_utils.get_ndim_from_sim(sim0)
        source_type = sim0.dtype

        operation = params['operation']
        method = params.get('method', '').lower()
        use_rotation = params.get('use_rotation', False)
        use_orthogonal_pairs = params.get('use_orthogonal_pairs', False)

        is_stack = ('stack' in operation)

        reg_channel = params.get('channel', 0)
        if isinstance(reg_channel, int):
            reg_channel_index = reg_channel
            reg_channel = None
        else:
            reg_channel_index = None

        if is_stack:
            # register in 2d; pairwise consecutive views
            register_sims = [si_utils.max_project_sim(sim, dim='z') for sim in register_sims]
            pairs = [(index, index + 1) for index in range(len(register_sims) - 1)]
        elif use_orthogonal_pairs:
            origins = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in register_sims])
            size = get_sim_physical_size(sim0)
            pairs, _ = get_orthogonal_pairs(origins, size)
            logging.info(f'#pairs: {len(pairs)}')
        else:
            pairs = None

        if 'cpd' in method:
            from src.registration_methods.RegistrationMethodCPD import RegistrationMethodCPD
            registration_method = RegistrationMethodCPD(source_type)
            pairwise_reg_func = registration_method.registration
        elif 'feature' in method:
            if 'cv' in method:
                from src.registration_methods.RegistrationMethodCvFeatures import RegistrationMethodCvFeatures
                registration_method = RegistrationMethodCvFeatures(source_type)
            else:
                from src.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures
                registration_method = RegistrationMethodSkFeatures(source_type)
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

        if self.verbose:
            progress = tqdm(desc='Registering', total=1)

        with ProgressBar() if self.verbose_mvs else nullcontext():
            try:
                logging.info('Registering...')
                register_msims = [msi_utils.get_msim_from_sim(sim) for sim in register_sims]
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

                # set missing transforms
                for sim in sims:
                    if self.reg_transform_key not in si_utils.get_tranform_keys_from_sim(sim):
                        si_utils.set_sim_affine(
                            sim,
                            param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                            transform_key=self.reg_transform_key)

                mappings = reg_result['params']
                # re-index from subset of sims
                residual_error_dict = reg_result.get('groupwise_resolution', {}).get('metrics', {}).get('residuals', {})
                residual_error_dict = {(indices[key[0]], indices[key[1]]): value.item()
                                       for key, value in residual_error_dict.items()}
                registration_qualities_dict = reg_result.get('pairwise_registration', {}).get('metrics', {}).get('qualities', {})
                registration_qualities_dict = {(indices[key[0]], indices[key[1]]): value
                                               for key, value in registration_qualities_dict.items()}
            except NotEnoughOverlapError:
                logging.warning('Not enough overlap')
                reg_result = {}
                mappings = [param_utils.identity_transform(ndim=ndims, t_coords=[0])] * len(sims)
                residual_error_dict = {}
                registration_qualities_dict = {}

        # re-index from subset of sims
        mappings_dict = {index: mapping for index, mapping in zip(indices, mappings)}

        if self.verbose:
            progress.update()
            progress.close()

        if is_stack:
            # set 3D affine transforms from 2D registration params
            for index, sim in enumerate(sims):
                affine_3d = param_utils.identity_transform(ndim=3)
                affine_3d.loc[{dim: mappings[index].coords[dim] for dim in mappings[index].sel(t=0).dims}] = mappings[index].sel(t=0)
                si_utils.set_sim_affine(sim, affine_3d, transform_key=self.reg_transform_key)

        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'residual_errors': residual_error_dict,
                'registration_qualities': registration_qualities_dict,
                'sims': sims}

    def fuse(self, sims, params):
        operation = params['operation']
        extra_metadata = params.get('extra_metadata', {})
        channels = extra_metadata.get('channels', [])
        z_scale = extra_metadata.get('scale', {}).get('z')
        if z_scale is None:
            if 'z' in sims[0].dims:
                z_scale = np.min(np.diff([si_utils.get_origin_from_sim(sim)['z'] for sim in sims]))
            else:
                z_scale = 1

        is_stack = ('stack' in operation)
        is_channel_overlay = (len(channels) > 1)

        sim0 = sims[0]
        source_type = sim0.dtype
        output_chunksize = {'y': 1024, 'x': 1024}
        for dim in sim0.dims:
            if dim not in output_chunksize:
                output_chunksize[dim] = 1

        if self.verbose:
            progress = tqdm(desc='Fusing registered', total=1)

        if is_stack:
            output_stack_properties = calc_output_properties(sims, self.reg_transform_key, z_scale=z_scale)
            # set z shape which is wrongly calculated by calc_stack_properties_from_view_properties_and_params
            # because it does not take into account the correct input z spacing because of stacks of one z plane
            output_stack_properties['shape']['z'] = len(sims)
            if self.verbose:
                logging.info(f'Output stack: {output_stack_properties}')

            data_size = np.prod(list(output_stack_properties['shape'].values())) * source_type.itemsize
            logging.info(f'Fusing Z stack {print_hbytes(data_size)}')

            # fuse all sims together using simple average fusion
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
            if self.verbose:
                logging.info(f'Output stack: {output_stack_properties}')
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
            if self.verbose:
                logging.info(f'Output stack: {output_stack_properties}')
            data_size = np.prod(list(output_stack_properties['shape'].values())) * source_type.itemsize
            logging.info(f'Fusing {print_hbytes(data_size)}')

            fused_image = fusion.fuse(
                sims,
                transform_key=self.reg_transform_key,
                output_chunksize=output_chunksize,
            )

        if self.verbose:
            progress.update()
            progress.close()

        return fused_image

    def calc_metrics(self, results):
        mappings = results['mappings']
        distances = [np.linalg.norm(param_utils.translation_from_affine(mapping.data[0]))
                     for mapping in mappings.values()]

        if len(distances) > 2:
            # Coefficient of variation
            cvar = np.std(distances) / np.mean(distances)
            confidence = 1 - min(cvar / 10, 1)
        else:
            size = get_sim_physical_size(results['sims'][0])
            norm_distance = np.sum(distances) / np.linalg.norm(size)
            confidence = 1 - min(math.sqrt(norm_distance), 1)

        residual_errors = results['residual_errors']
        if len(residual_errors) > 0:
            residual_error = np.mean(list(residual_errors.values()))
        else:
            residual_error = 1

        registration_qualities = {key: value.item() for key, value in results['registration_qualities'].items()}
        if len(registration_qualities) > 0:
            registration_quality = np.mean(list(registration_qualities.values()))
        else:
            registration_quality = 0

        summary = (f'Residual error: {residual_error:.3f}'
                   f' Registration quality: {registration_quality:.3f}'
                   f' Confidence: {confidence:.3f}')

        return {'confidence': confidence,
                'residual_error': residual_error,
                'residual_errors': residual_errors,
                'registration_quality': registration_quality,
                'registration_qualities': registration_qualities,
                'summary': summary}
