# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import copy
import dask
from dask.diagnostics import ProgressBar
from enum import Enum, auto
import logging
from multiview_stitcher import registration, vis_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from multiview_stitcher.param_resolution import groupwise_resolution
from multiview_stitcher.registration import compute_pairwise_registrations, _plot_registration_summaries
import networkx as nx
import numpy as np
import os.path
from pathlib import Path
import shutil
from skimage.transform import resize
import xarray as xr

from muvis_align.constants import *
from muvis_align.file.rocrate_utils import create_ro_crate, create_zarr_ro_crate
from muvis_align.file.transforms import write_transforms, read_transforms
from muvis_align.image.Video import Video
from muvis_align.image.flatfield import flatfield_correction
from muvis_align.image.ome_helper import save_image
from muvis_align.image.ome_zarr_helper import save_ome_multiscale_levels
from muvis_align.image.ome_tiff_helper import save_tiff
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import *
from muvis_align.metrics import calc_pair_metrics, calc_global_metrics, quality_to_scalar
from muvis_align.Timer import Timer
from muvis_align.util import *


class RegState(Enum):
    UNINIT = auto()
    INIT = auto()
    SIMS_INIT = auto()
    PAIRS_REG = auto()
    GLOBAL_REG = auto()
    FUSED = auto()


class MVSRegistration:
    def __init__(self, operation='register', label='', input_path=None, output_path=None,
                 source_metadata={}, extra_metadata={},
                 global_rotation=None, global_center=None,
                 overwrite=True, clear=False, ui='', verbose=False, debug=False):
        # set here (rather than only in init(), below) so verbose/logging_dask/logging_time are
        # always defined, even for an instance that never gets init() called on it (e.g. Interface's
        # placeholder self.reg before a project is loaded)
        self.verbose = verbose
        self.logging_dask = self.verbose
        self.logging_time = self.verbose
        self.reset()

        if input_path is not None:
            self.init(operation=operation, label=label, input_path=input_path, output_path=output_path,
                      source_metadata=source_metadata, extra_metadata=extra_metadata,
                      global_rotation=global_rotation, global_center=global_center,
                      overwrite=overwrite, clear=clear, ui=ui, verbose=verbose, debug=debug)

    @property
    def msims(self):
        # built lazily from the cheap per-source geometry init_data() already resolved (sources/
        # positions/_msim_transforms/_msim_output_order/_msim_z_scale) - real msim construction
        # (the expensive part) is deferred until something genuinely needs pixel-shaped data
        # (preview fusion, pre-processing/registration/fusion), not eagerly for every project load
        return self.ensure_msims()

    @msims.setter
    def msims(self, value):
        self._msims = value

    def ensure_msims(self, progress_factory=None):
        # same lazy build the msims property triggers, but callable ahead of time with a
        # progress_factory - lets a caller that's about to force this (e.g. run_pre_processing())
        # show fine-grained, per-source progress for it instead of it happening silently as a
        # side effect of evaluating `self.msims` as a plain argument expression
        if self._msims is None:
            self._build_msims(progress_factory=progress_factory)
        return self._msims

    def _build_msims(self, progress_factory=None):
        progress_context = (
            progress_factory(total=len(self.sources), desc='Building sources')
            if progress_factory is not None
            else nullcontext(None)
        )
        with progress_context as pbar:
            msims = []
            for source, translation, transform in zip(self.sources, self.positions, self._msim_transforms):
                msims.append(build_source_msim(source, self._msim_output_order, translation, transform,
                                               self.source_transform_key, z_scale=self._msim_z_scale))
                if pbar is not None:
                    pbar.update(1)
        self._msims = msims

    def reset(self):
        self.state = RegState.UNINIT
        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'
        self.source_metadata = {}
        self.extra_metadata = {}
        self.msims = []
        self.register_msims = None
        self.sources = []
        self.metrics = {}
        self.register_indices = None
        self.output_params = {}

    def is_initialised(self):
        return self.state.value >= RegState.INIT.value

    def is_pairs_registered(self):
        return self.state.value >= RegState.PAIRS_REG.value

    def is_global_registered(self):
        return self.state.value >= RegState.GLOBAL_REG.value

    def is_fused(self):
        return self.state.value >= RegState.FUSED.value

    def init_params(self, params_general, params, label='', input_path=None, global_rotation=None, global_center=None):
        self.params_general = params_general
        self.params = params
        self.input_params = params.get('input')
        if isinstance(self.input_params, (str, list)):
            self.input_params = {'path': self.input_params}
        if input_path is None:
            input_path = self.input_params.get('path')
        self.output_params = params.get('output')
        if isinstance(self.output_params, str):
            self.output_params = {'path': self.output_params}
        self.preprocess_params = params.get('preprocessing', {})
        self.register_params = params.get('registration', {})
        self.fusion_params = params.get('fusion', {})

        return self.init(operation=params.get('operation'), label=label, input_path=input_path,
                         input_labels=self.input_params.get('labels'),
                         output_path=self.output_params.get('path'),
                         source_metadata=self.input_params.get('source_metadata', {}),
                         extra_metadata=self.input_params.get('extra_metadata', {}),
                         global_rotation=global_rotation, global_center=global_center,
                         overwrite=params_general.get('overwrite', False), clear=params_general.get('clear', False),
                         ui=params_general.get('ui', ''),
                         verbose=params_general.get('verbose', False), debug=params_general.get('debug', False))

    def init(self, operation='', label='', input_path=None, input_labels=None, output_path=None,
             source_metadata={}, extra_metadata={}, global_rotation=None, global_center=None,
             overwrite=True, clear=False, ui='', verbose=False, debug=False):
        self.overwrite = overwrite
        self.clear = clear
        self.ui = ui
        self.verbose = verbose
        self.debug = debug
        self.logging_dask = self.verbose
        self.logging_time = self.verbose
        self.mpl_ui = ('mpl' in self.ui or 'plot' in self.ui)
        self.operation = operation
        self.fileset_label = label
        self.global_rotation = global_rotation
        self.global_center = global_center
        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'
        self.msims = []
        self.sources = []
        self.state = RegState.INIT

        self.input_path = input_path
        if isinstance(input_path, list):
            self.filenames = input_path
            self.input_dir = os.path.dirname(input_path[0])
        elif os.path.isdir(input_path):
            self.filenames = dir_regex(os.path.join(input_path, '*'))
            self.input_dir = input_path
        else:
            self.filenames = dir_regex(input_path)
            self.input_dir = os.path.dirname(input_path)
        if not self.filenames:
            return False

        self.filenames = [Path(path).as_posix() for path in self.filenames]

        if input_labels:
            self.file_labels = input_labels
        else:
            self.file_labels = get_unique_file_labels(self.filenames)

        self.source_metadata = source_metadata
        self.extra_metadata = extra_metadata

        try:
            output_path = output_path.format_map(split_numeric_dict(self.filenames[0]))
            self.output = os.path.join(self.input_dir, output_path)    # preserve trailing slash: do not use os.path.normpath()
            output_dir = os.path.dirname(self.output)
            if self.clear:
                shutil.rmtree(output_dir, ignore_errors=True)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Error initializing output directory: {e}")
            return False

        return True

    def run(self):
        with ProgressBar(minimum=60, dt=1) if self.logging_dask else nullcontext():
            return self._run()

    def _run(self):
        filenames = self.filenames
        file_labels = self.file_labels

        output = self.output
        operation = self.operation
        source_metadata = self.source_metadata
        extra_metadata = self.extra_metadata
        if isinstance(extra_metadata, dict):
            z_scale = extra_metadata.get('scale', {}).get('z')
            channels = extra_metadata.get('channels', [])
        else:
            z_scale = None
            channels = []
        normalise_orientation = 'norm' in source_metadata
        output_params = self.output_params
        general_output_params = self.params_general.get('output', {})
        overlap_threshold = self.register_params.get('overlap_threshold', self.params.get('overlap_threshold', 0.5))
        save_images = self.output_params.get('save_images', self.params.get('save_images', True))

        output_format = output_params.get('format', general_output_params.get('format', zarr_extension))
        output_tile_size = output_params.get('tile_size', general_output_params.get('tile_size'))
        output_compression = output_params.get('compression', general_output_params.get('compression'))
        output_pyramid_downsample = output_params.get('pyramid_downsample', general_output_params.get('pyramid_downsample', 2))
        output_npyramid_add = output_params.get('npyramid_add', general_output_params.get('npyramid_add', 0))
        output_ome_version = output_params.get('ome_version', general_output_params.get('ome_version', default_ome_zarr_version))

        if len(filenames) == 0:
            logging.warning('Skipping (no images)')
            return False

        output_filename = operation_to_past_participle(operation)

        self.check_progress(output_filename, output_format)

        if self.is_fused() and not self.overwrite:
            logging.warning(f'Skipping existing output {output_filename}')
            return False

        with Timer('init sims', self.logging_time):
            self.init_data()
            msims = self.msims

        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        is_stack = ('stack' in operation)
        is_simple_stack = is_stack and not is_3d
        is_transition = ('transition' in operation)
        is_channel_overlay = (len(channels) > 1)
        if not z_scale:
            z_scale = self.scales[0].get('z', 1)

        with Timer('pre-process', self.logging_time):
            # preprocess() is msims-in/msims-out - register() itself is msims-based too now
            self.preprocess(msims, **self.preprocess_params)
            register_msims, register_indices = self.register_msims, self.register_indices

        self.init_progress(output_filename, output_format)

        data = []
        mappings_header = ['id', 'filename', 'x_pixels', 'y_pixels', 'z_pixels', 'x', 'y', 'z', 'rotation']
        for label, filename, position, rotation, scale in zip(file_labels, self.filenames, self.positions, self.rotations, self.scales):
            position_pixels = {dim: position[dim] / float(scale.get(dim, 1)) for dim in position.keys()}
            row = [label] + [filename] + dict_to_xyz(position_pixels, add_zeros=True) + dict_to_xyz(position, add_zeros=True) + [rotation]
            data.append(row)
        export_csv(output + prereg_mappings_name, data, header=mappings_header)

        if len(filenames) == 1 and save_images and not 'register' in operation and not 'stack' in operation:
            logging.warning('Skipping operation (single image)')
            sim = msi_utils.get_sim_from_msim(msims[0], scale='scale0')
            self.save(output_filename, sim, translations0=self.positions,
                      format=output_format,
                      tile_size=output_tile_size,
                      pyramid_downsample=output_pyramid_downsample,
                      npyramid_add=output_npyramid_add,
                      ome_version=output_ome_version)
            return False

        _, has_overlaps = self.validate_overlap(msims, file_labels, is_stack=is_simple_stack,
                                                expect_large_overlap=is_simple_stack or is_channel_overlay)
        overall_overlap = np.mean(has_overlaps)
        if overall_overlap < overlap_threshold:
            raise ValueError(f'Not enough overlap: {overall_overlap * 100:.1f}%')

        if not self.is_global_registered() or self.overwrite:
            if 'register' in operation:
                with Timer('register', self.logging_time):
                    results = self.register(register_msims, register_indices, self.register_params)
                reg_result = results['reg_result']
                mappings = results['mappings']
                metrics = results['metrics']

            if is_stack:
                msims = make_msims_3d(msims, z_scale, self.positions)

            if 'register' in operation:
                logging.info(metrics['summary'])
                self.save_mappings_csv(mappings, normalise_orientation=normalise_orientation)

                for reg_label, reg_item in reg_result.items():
                    if isinstance(reg_item, dict):
                        summary_plot = reg_item.get('summary_plot')
                        if summary_plot is not None:
                            figure, axes = summary_plot
                            summary_plot_filename = output + f'{reg_label}.pdf'
                            figure.savefig(summary_plot_filename)

        self.msims = msims
        registered_positions_filename = output + registered_positions_name
        if self.reg_transform_key in get_msim_transform_keys(msims[0]):
            transform_key = self.reg_transform_key
            with Timer('plot positions', self.logging_time):
                # plot_positions is a library function that needs concrete sims - derived here on
                # demand from msims (the registered transform lives there now, see register_global)
                plot_sims = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims]
                vis_utils.plot_positions(plot_sims, transform_key=transform_key,
                                         use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                         show_plot=self.mpl_ui, output_filename=registered_positions_filename)
                plt_close()
        else:
            transform_key = self.source_transform_key

        logging.info('Exporting...')

        image_paths = []
        if save_images:
            if self.output_params.get('preview'):
                with Timer('create preview', self.logging_time):
                    self.create_preview('preview_' + output_filename,
                                        nom_msims=msims,
                                        transform_key=transform_key)

            if 'register' in operation or 'stack' in operation or 'fuse' in operation:
                with Timer('fuse image', self.logging_time):
                    if isinstance(self.fusion_params, dict):
                        fusion_method = self.fusion_params.get('method', '')
                        output_spacing = self.fusion_params.get('output_spacing', 'mean')
                    else:
                        fusion_method = self.fusion_params
                        output_spacing = self.params.get('output_spacing', 'mean')
                    zarr_output_filename = output_filename if 'zar' in output_format else None
                    # msims was z-stacked in lockstep with sims above (make_msims_3d) when
                    # is_stack, so it's always safe to hand to fuse() as the primary input here
                    fused_msim, is_saved = self.fuse(msims, fusion_method=fusion_method, output_spacing=output_spacing,
                                                     transform_key=transform_key, output_filename=zarr_output_filename,
                                                     tile_size=output_tile_size, ome_version=output_ome_version)
                    self.state = RegState.FUSED
            else:
                fused_msim = msims
                is_saved = False

            # save_image() (and save_video(), for a transition) only accept sims - extract them
            # from the fused msim(s) here, once, right at the point they're actually needed
            fused_sims = extract_sims_from_fused(fused_msim)

            if not is_saved or 'tif' in output_format:
                extra_output_format = output_format
                if is_saved:
                    extra_output_format = extra_output_format.replace('ome.zarr', '').replace('zar', '')
                logging.info('Saving fused image...')
                with Timer('save fused image', self.logging_time):
                    self.save(output_filename, fused_sims,
                              transform_key=transform_key, translations0=self.positions,
                              format = extra_output_format,
                              tile_size = output_tile_size,
                              compression = output_compression,
                              pyramid_downsample = output_pyramid_downsample,
                              npyramid_add = output_npyramid_add,
                              ome_version = output_ome_version)

            if 'tif' in output_format:
                filename = output_filename + tiff_extension
                image_paths.append(filename)
            if 'zar' in output_format:
                filename = output_filename + zarr_extension
                image_paths.append(filename)
                create_zarr_ro_crate(self.output + filename)

        create_ro_crate(fused_msim, self.output, image_paths)

        if is_transition:
            self.save_video(output, msims, fused_msim)

        return True

    def init_sources(self, progress_factory=None):
        source_metadata0 = self.source_metadata
        source_metadata = {}
        nfiles = len(self.filenames)
        self.sources = [None] * nfiles
        progress_context = (
            progress_factory(total=nfiles, desc='Initialising sources')
            if progress_factory is not None
            else nullcontext(None)
        )
        # resolve each file's own source_metadata up front (cheap, in-memory) - `source_metadata`
        # is reused/mutated across iterations below (matching the previous sequential behaviour),
        # so each file gets its own snapshot rather than a shared dict that the source
        # construction below (parallelised, so no longer necessarily reading it immediately) could
        # see mutated to a later file's values
        per_file_metadata = []
        for index, label in enumerate(self.file_labels):
            if isinstance(source_metadata0, dict) and label in source_metadata0:
                source_metadata = source_metadata0[label]
                position, rotation, scale = get_properties_from_transform(param_utils.affine_to_xaffine(np.array(source_metadata)))
                source_metadata = {'position': position, 'rotation': rotation, 'scale': xyz_to_dict([scale, scale])}
            else:
                if 'position' in source_metadata0:
                    translation = source_metadata0['position']
                    if isinstance(translation, list):
                        translation = translation[index]
                    source_metadata['position'] = translation
                if 'scale' in source_metadata0:
                    scale = source_metadata0['scale']
                    if isinstance(scale, list):
                        scale = scale[index]
                    source_metadata['scale'] = scale
                if 'rotation' in source_metadata0:
                    source_metadata['rotation'] = source_metadata0['rotation']
            if isinstance(source_metadata0, dict):
                # blanket per-run flags that apply identically to every source
                for flag in ('sbem', 'invert', 'is_center'):
                    if flag in source_metadata0:
                        source_metadata[flag] = source_metadata0[flag]
            per_file_metadata.append(copy.deepcopy(source_metadata))

        with progress_context as pbar:
            def build_source(index, matrix_size):
                return create_image_source(
                    self.filenames[index], per_file_metadata[index], extra_metadata=self.extra_metadata,
                    file_label=self.file_labels[index], transform_key=self.source_transform_key,
                    matrix_size=matrix_size)

            # matrix_size is decided once from the first source (matching the previous
            # is_3d-from-source0 behaviour) - build it on its own first, so every other source
            # below can be constructed with that already-known matrix_size from the start
            first_source = build_source(0, matrix_size=None)
            matrix_size = 4 if first_source.get_size().get('z', 0) > 1 else 3
            self.sources[0] = first_source
            if pbar is not None:
                pbar.update(1)

            if nfiles > 1:
                max_workers = min(default_source_init_workers, nfiles - 1)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(build_source, index, matrix_size): index
                              for index in range(1, nfiles)}
                    for future in as_completed(futures):
                        index = futures[future]
                        self.sources[index] = future.result()
                        if pbar is not None:
                            pbar.update(1)

    def init_data(self, source_metadata={}, extra_metadata={}, z_scale=None, target_scale=None, store=True,
                  progress_factory=None):
        if not source_metadata:
            source_metadata = self.source_metadata
        if not extra_metadata:
            extra_metadata = self.extra_metadata
        source_metadata = import_metadata(source_metadata, input_path=self.input_path)
        extra_metadata = import_metadata(extra_metadata, input_path=self.input_path)
        source_metadata_changed = (source_metadata != self.source_metadata)
        self.source_metadata = source_metadata
        self.extra_metadata = extra_metadata
        if isinstance(source_metadata, dict):
            z_scale = source_metadata.get('scale', {}).get('z')
        if not z_scale and isinstance(extra_metadata, dict):
            z_scale = extra_metadata.get('scale', {}).get('z')

        if len(self.filenames) == 0:
            raise ValueError('No input files')

        logging.info('Initialising sims...')
        if not self.sources or source_metadata_changed:
            self.init_sources(progress_factory=progress_factory)
        sources = self.sources
        source0 = sources[0]
        sims = []
        scales = []
        translations = []
        rotations = []
        levels = []
        rescales = []

        is_3d = (source0.get_size().get('z', 0) > 1)
        is_stack = ('stack' in self.operation)
        output_order = 'zyx' if is_3d else 'yx'

        ndims = len(output_order)
        if source0.get_nchannels() > 1:
            if source0.is_rgb:
                output_order = output_order + 'c'
            else:
                output_order = 'c' + output_order

        last_z_position = None
        different_z_positions = False
        delta_zs = []
        for filename, source in zip(self.filenames, sources):
            # position/scale/rotation, and per-source corrections (SBEM, is_center, invert),
            # are now resolved by ImageSource itself (see ImageSource.fix_metadata) - this pass
            # reads only cached metadata off `source`, no pixel data is touched
            scale = source.get_pixel_size()
            translation = source.get_position()
            rotation = source.get_rotation()

            level = 0
            rescale = {}
            if target_scale:
                # Only downscaling
                level, rescale, scale = get_level_from_scale(source, target_scale)
            if 'z' in translation:
                z_position = translation['z']
            else:
                z_position = 0
            if last_z_position is not None and z_position != last_z_position:
                delta_zs.append(z_position - last_z_position)
            if 'rotation' in source_metadata:
                rotation = source_metadata['rotation']
            if self.global_rotation is not None:
                rotation = self.global_rotation

            scales.append(scale)
            translations.append(translation)
            rotations.append(rotation)
            levels.append(level)
            rescales.append(rescale)
            last_z_position = z_position

        if 'z' in output_order and z_scale is None:
            if len(delta_zs) > 0:
                z_scale = np.min(delta_zs)
            else:
                z_scale = 1

        if 'norm' in source_metadata:
            sizes = [source.get_physical_size() for source in sources]
            center = {dim: 0 for dim in output_order}
            if 'center' in source_metadata:
                if 'global' in source_metadata:
                    center = self.global_center
                else:
                    center = {dim: float(np.mean([translation[dim] for translation in translations])) for dim in translations[0]}
            translations, rotations = normalise_rotated_positions(translations, rotations, sizes, center, len(output_order))

        #translations = [np.array(translation) * 1.25 for translation in translations]

        increase_z_positions = is_stack and not different_z_positions

        z_position = 0
        final_scales = []
        final_translations = []
        transforms = []
        msims = [] if not store else None
        for source, level, rescale, scale, translation, rotation, file_label in zip(
                sources, levels, rescales, scales, translations, rotations, self.file_labels):
            # transform #dimensions need to match
            if 'z' in output_order:
                if len(scale) > 0 and 'z' not in scale:
                    scale['z'] = abs(z_scale)
                if (len(translation) > 0 and 'z' not in translation) or increase_z_positions:
                    translation['z'] = z_position
                if increase_z_positions:
                    z_position += z_scale
            if rotation is None or 'norm' in source_metadata:
                # if positions are normalised, don't use rotation
                transform = None
            else:
                transform = param_utils.invert_coordinate_order(
                    create_transform(translation, rotation, matrix_size=ndims + 1)
                )
            if file_label in extra_metadata:
                transform2 = extra_metadata[file_label]
                if transform is None:
                    transform = np.array(transform2)
                else:
                    transform = np.array(combine_transforms([transform, transform2]))

            if translation:
                if 'x' not in translation:
                    translation['x'] = 0
                if 'y' not in translation:
                    translation['y'] = 0

            transforms.append(transform)
            if not store:
                # build this source's own multiscale msim directly from its already-correct,
                # cached msim (ImageSource.get_msim, itself built from fix_metadata/_build_msim/
                # _restamp_msim) - only the run-level deltas that no single source can know about
                # itself (cross-source normalisation, z-stacking, extra_metadata) are applied
                # here, via assign_coords, never by reconstructing from raw arrays with
                # si_utils.get_sim_from_array
                msim = build_source_msim(source, output_order, translation, transform, self.source_transform_key,
                                         z_scale=z_scale)
                msims.append(msim)
            final_scales.append(scale)
            final_translations.append(translation)

        if store:
            self.scales = final_scales
            self.positions = final_translations
            self.rotations = rotations
            self.state = RegState.SIMS_INIT
            # defer the actual per-source msim build (build_source_msim(), the expensive part)
            # until self.msims is genuinely read - see the msims property/_build_msims()
            self._msim_output_order = output_order
            self._msim_z_scale = z_scale
            self._msim_transforms = transforms
            self._msims = None
            return None

        return msims

    def check_progress(self, output_filename, output_format):
        pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
        mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
        if self.output_exists(output_filename, output_format):
            self.state = RegState.FUSED
        elif os.path.exists(mappings_filename):
            self.state = RegState.GLOBAL_REG
        elif os.path.exists(pair_mappings_filename):
            self.state = RegState.PAIRS_REG

    def init_progress(self, output_filename, output_format):
        pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
        mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
        metrics_filename = self.output + metrics_name
        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        self.check_progress(output_filename, output_format)

        if self.is_pairs_registered() and os.path.exists(pair_mappings_filename):
            # load pair mapping and initialise pair_graph
            logging.info(f'Loading pair mapping from {pair_mappings_filename}')
            pairs = import_json(pair_mappings_filename)
            indexed_pair_transforms = {}
            indexed_qualities = {}
            indexed_bboxes = {}
            for key, value in pairs.items():
                key1, key2 = json.loads(key)
                index1, index2 = find_file_list_index(self.filenames, key1), find_file_list_index(self.filenames, key2)
                if index1 is not None and index2 is not None:
                    indexed_key = index1, index2
                    indexed_pair_transforms[indexed_key] = (
                        param_utils.affine_to_xaffine(np.array(value['mapping'])).expand_dims({'t': [0]}))
                    indexed_qualities[indexed_key] = np.array(value.get(default_quality_key, 0))
                    if 'bbox' in value:
                        indexed_bboxes[indexed_key] = xr.DataArray(value['bbox'])
            if not is_3d:
                self.msims = make_msims_2d(self.msims)
            self.pair_msims = self.msims
            self.pairs = list(indexed_pair_transforms.keys())
            self.metrics = {
                'summary': {default_transform_key: {default_quality_key: np.mean(list(indexed_qualities.values()))}},
                'pairs': {key: {default_transform_key: {default_quality_key: value.item()}}
                          for key, value in indexed_qualities.items()}
            }
            with dask.config.set(scheduler='single-threaded'):
                self.pairs_graph = mv_graph.build_view_adjacency_graph_from_msims(
                    self.pair_msims,
                    transform_key=self.source_transform_key,
                    pairs=self.pairs
                )
            nx.set_edge_attributes(self.pairs_graph, indexed_pair_transforms, default_transform_key)
            nx.set_edge_attributes(self.pairs_graph, indexed_qualities, default_quality_key)
            nx.set_edge_attributes(self.pairs_graph, indexed_bboxes, 'bbox')

        if self.is_global_registered():
            logging.info(f'Loading global mapping from {mappings_filename}')

            is_stack = ('stack' in self.operation)
            #z_positions = set([source.get_position().get('z', 0) for source in self.sources])
            #make_3d = len(z_positions) > 1 or is_stack
            make_3d = is_stack
            if isinstance(self.extra_metadata, dict):
                z_scale = self.extra_metadata.get('scale', {}).get('z')
            else:
                z_scale = None

            mappings = read_transforms(mappings_filename)
            # write reg_transform_key onto self.msims (msim -> msim, every scale, no sim needed) -
            # the persistent pyramid needs the same transform a fresh registration run would have
            # written via register_global, or copy_transforms/get_transforms downstream
            # (Interface.py) won't find it there when resuming from saved state
            for msim, filename in zip(self.msims, self.filenames):
                mapping = param_utils.affine_to_xaffine(np.array(find_file_dict_item(mappings, filename)))
                if make_3d:
                    transform = param_utils.identity_transform(ndim=3)
                    transform.loc[{dim: mapping.coords[dim] for dim in mapping.dims}] = mapping
                else:
                    transform = mapping
                msi_utils.set_affine_transform(msim, transform, transform_key=self.reg_transform_key)
            if make_3d:
                self.msims = make_msims_3d(self.msims, z_scale, self.positions)
            elif not is_3d:
                self.msims = make_msims_2d(self.msims)
            self.pair_msims = self.msims
            metrics = import_json(metrics_filename)
            indexed_metrics = {}
            for key, value in metrics.items():
                key1, key2 = json.loads(key)
                index1, index2 = find_file_list_index(self.filenames, key1), find_file_list_index(self.filenames, key2)
                if index1 is not None and index2 is not None:
                    indexed_key = index1, index2
                    indexed_metrics[indexed_key] = value
            self.metrics = {
                'summary': {default_transform_key:
                                {self.reg_transform_key: np.mean([value[default_quality_key]
                                                                  for value in indexed_metrics.values()
                                                                  if default_quality_key in value])}},
                'pairs': {key: {self.reg_transform_key: value} for key, value in indexed_metrics.items()}
            }

    def validate_overlap(self, sims, labels, is_stack=False, expect_large_overlap=False):
        # accepts either sims or msims (each msim's scale0 sim is used) - only position/size
        # metadata is ever read here, never pixel data
        sims = [msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
               for item in sims]
        min_dists = []
        has_overlaps = []
        n = len(sims)
        positions = [get_sim_position_final(sim, get_center=True) for sim in sims]
        sizes = [float(np.linalg.norm(list(get_sim_physical_size(sim).values()))) for sim in sims]
        for i in range(n):
            norm_dists = []
            # check if only single z slices
            if is_stack:
                if i + 1 < n:
                    compare_indices = [i + 1]
                else:
                    compare_indices = []
            else:
                compare_indices = range(n)
            for j in compare_indices:
                if not j == i:
                    distance = math.dist(positions[i].values(), positions[j].values())
                    norm_dist = distance / np.mean([sizes[i], sizes[j]])
                    norm_dists.append(norm_dist)
            if len(norm_dists) > 0:
                norm_dist = min(norm_dists)
                min_dists.append(float(norm_dist))
                if norm_dist >= 1:
                    logging.warning(f'{labels[i]} has no overlap')
                    has_overlaps.append(False)
                elif expect_large_overlap and norm_dist > 0.5:
                    logging.warning(f'{labels[i]} has small overlap')
                    has_overlaps.append(False)
                else:
                    has_overlaps.append(True)
        return min_dists, has_overlaps

    def preprocess(self, msims, scale=None,
                   flatfield_quantiles=None, normalisation=None, gaussian_sigma=None, filter_foreground=False,
                   progress_factory=None,
                   **kwargs):
        def normalisation_enabled(value):
            if isinstance(value, str) and value.lower() in ['false', 'no', 'none', '']:
                return False
            return bool(value)

        do_normalisation = normalisation_enabled(normalisation)

        def count_progress_steps():
            n_steps = 0
            if scale and scale != 1:
                n_steps += 1
            if filter_foreground:
                # foreground map + final foreground filtering
                n_steps += 2
            if flatfield_quantiles:
                n_steps += 1
            if gaussian_sigma:
                n_steps += 1
            if do_normalisation:
                n_steps += 1
            return max(n_steps, 1)

        def update_progress():
            if pbar is not None:
                pbar.update(1)

        progress_context = (
            progress_factory(total=count_progress_steps(), desc='Pre-processing')
            if progress_factory is not None
            else nullcontext(None)
        )

        with progress_context as pbar:
            modified = False
            # normalise pixel size: take max pixel size
            max_scale = {dim: max(scale.get(dim, 1) for scale in self.scales) for dim in 'xy'}
            scales0 = self.scales

            if scale and scale != 1:
                # select every native level at or coarser than the requested scale, as a real
                # (smaller) sub-pyramid, rather than resizing to one exact resolution (unnecessary
                # for registration - "coarse enough" is all it needs, and it can auto-select its
                # own resolution from whatever real levels it's handed - see register_pairs)
                msims = select_msim_subpyramid_at_scale(msims, self.sources, scale)
                modified = True
                update_progress()

            if filter_foreground:
                foreground_map = calc_foreground_map(msims)
                modified = True
                update_progress()
            else:
                foreground_map = None

            if flatfield_quantiles:
                logging.info('Flat-field correction...')
                if isinstance(flatfield_quantiles, str):
                    flatfield_quantiles = [float(quantile.strip()) for quantile in flatfield_quantiles.split(',')]
                new_msims = [None] * len(msims)
                for msim_indices in group_sims_by_z(msims, self.positions):
                    msims_z_set = [msims[i] for i in msim_indices]
                    foreground_map_z_set = [foreground_map[i] for i in
                                            msim_indices] if foreground_map is not None else None
                    # flatfield_correction is msims-in/msims-out: the correction model (quantile
                    # images) is computed once per z-set at scale0, then resized to every other
                    # pyramid level - the whole msim ends up corrected, not just scale0
                    new_msims_z_set = flatfield_correction(msims_z_set, self.source_transform_key,
                                                           flatfield_quantiles,
                                                           foreground_map=foreground_map_z_set)
                    for msim_index, msim in zip(msim_indices, new_msims_z_set):
                        new_msims[msim_index] = msim
                msims = new_msims
                modified = True
                update_progress()

            if gaussian_sigma:
                logging.info('Applying Gaussian filtering...')

                def sigma_for_level(level_sim, scale0, sigma0):
                    # keep the same physical blur radius at every resolution: scale the notional
                    # level-0 pixel-space sigma by how much coarser this level's pixels are - a
                    # no-op (ratio 1) at native level0's own resolution
                    level_scale = si_utils.get_spacing_from_sim(level_sim)
                    ratios = [scale0[dim] / level_scale[dim] for dim in scale0 if dim in level_scale]
                    return sigma0 * (np.mean(ratios) if ratios else 1)

                new_msims = []
                for msim, scale0 in zip(msims, scales0):
                    # factor in original pixel size for gaussian sigma value
                    rel_scale = np.mean(list(scale0.values())) / np.mean(list(max_scale.values()))
                    sigma0 = gaussian_sigma * (rel_scale ** (1 / 3))

                    def level_func(level_sim, scale_key, scale0=scale0, sigma0=sigma0):
                        sigma = sigma_for_level(level_sim, scale0, sigma0)
                        return gaussian_filter_sim(level_sim, self.source_transform_key, sigma)
                    new_msims.append(map_msim_levels(msim, level_func))
                msims = new_msims
                modified = True
                update_progress()

            if do_normalisation:
                use_global = ('global' in str(normalisation).lower())
                if use_global:
                    logging.info('Normalising (global)...')
                else:
                    logging.info('Normalising (individual)...')
                norm_stats, norm_dtype = calc_normalise_stats(msims, use_global=use_global)
                new_msims = []
                for msim, (norm_min, norm_range) in zip(msims, norm_stats):
                    # reuse the exact same min/range computed at the working resolution (scale0)
                    # at every pyramid level, rather than recomputing per level - keeps
                    # normalisation constants identical across the whole pyramid
                    def level_func(level_sim, scale_key, norm_min=norm_min, norm_range=norm_range):
                        return normalise_sim(level_sim, self.source_transform_key, norm_min, norm_range, norm_dtype)
                    new_msims.append(map_msim_levels(msim, level_func))
                msims = new_msims
                modified = True
                update_progress()

            if filter_foreground:
                logging.info('Filtering foreground images...')
                # tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
                # threshold1 = np.mean(tile_vars)
                # threshold2 = np.median(tile_vars)
                # threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
                # threshold = min(threshold1, threshold2, threshold3)
                # foregrounds = (tile_vars >= threshold)
                new_msims = [msim for msim, is_foreground in zip(msims, foreground_map) if is_foreground]
                logging.info(f'Foreground images: {len(new_msims)} / {len(msims)}')
                indices = np.where(foreground_map)[0]
                msims = new_msims
                modified = True
                update_progress()
            else:
                indices = range(len(msims))

        self.register_msims = msims
        self.register_indices = indices
        return msims, indices, modified

    def create_registration_method(self, sim0, params={}, method=''):
        registration_method = None
        pairwise_reg_func_kwargs = None

        if 'registration' in params:
            params = params['registration']
        if not method:
            method = params.get('method',
                                params.get('name', ''))
        method = method.lower()

        if 'cpd' in method:
            from muvis_align.registration_methods.RegistrationMethodCPD import RegistrationMethodCPD
            registration_method = RegistrationMethodCPD(sim0, params, self.debug)
            pairwise_reg_func = registration_method.registration
        elif 'feature' in method or 'orb' in method or 'sift' in method:
            if 'cv' in method:
                from muvis_align.registration_methods.RegistrationMethodCvFeatures import RegistrationMethodCvFeatures
                registration_method = RegistrationMethodCvFeatures(sim0, params, self.debug)
            else:
                from muvis_align.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures
                registration_method = RegistrationMethodSkFeatures(sim0, params, self.debug)
            pairwise_reg_func = registration_method.registration
        elif 'elastix' in method:
            try:
                from itk import ElastixRegistrationMethod
            except ImportError:
                raise ImportError('ITK-Elastix is required for ITK-Elastix registration.')
            pairwise_reg_func = registration.registration_ITKElastix
        elif 'ant' in method:
            try:
                import ants
            except ImportError:
                raise ImportError('ANTsPy is required for ANTsPy registration.')
            pairwise_reg_func = registration.registration_ANTsPy
            # args for ANTsPy registration: used internally by ANYsPy algorithm
            pairwise_reg_func_kwargs = {
                'transform_types': ['Rigid'],
                "aff_random_sampling_rate": 0.5,
                "aff_iterations": (2000, 2000, 1000, 1000),
                "aff_smoothing_sigmas": (4, 2, 1, 0),
                "aff_shrink_factors": (16, 8, 2, 1),
            }
        else:
            pairwise_reg_func = registration.phase_correlation_registration

        self.registration_method = registration_method

        return method, pairwise_reg_func, pairwise_reg_func_kwargs

    def select_pair_overlap(self, msim1, msim2, params=None):
        """Determine the resolution register_pair_of_msims_over_time would auto-select for this
        pair (using the same public multiview_stitcher helpers it uses internally:
        get_optimal_registration_binning + msi_utils.get_res_level_from_binning_factors) and
        extract the two sims' overlap region at that resolution. This crop depends only on the
        source data and the registration channel - never on the registration method or its
        tuning parameters - so a caller (e.g. a UI preview) can cache it across repeated
        parameter-only changes, instead of re-selecting resolution and re-cropping from the
        (possibly large) source data every time. See register_overlap() to actually register on
        the returned crop.
        """
        params = params or {}
        # select a single registration channel first - get_optimal_registration_binning/
        # get_overlap_images below have no 'c' dim handling of their own, unlike register_pairs()'s
        # own channel-selection (which only runs as part of its full multi-pair graph, not for a
        # single ad-hoc pair)
        if 'c' in get_msim_dims(msim1):
            reg_channel = params.get('channel', 0)
            if isinstance(reg_channel, int):
                reg_channel = get_msim_image0(msim1).coords['c'][reg_channel]
            msim1 = msi_utils.multiscale_sel_coords(msim1, {'c': reg_channel})
            msim2 = msi_utils.multiscale_sel_coords(msim2, {'c': reg_channel})

        sim1_0 = msi_utils.get_sim_from_msim(msim1, scale='scale0')
        sim2_0 = msi_utils.get_sim_from_msim(msim2, scale='scale0')
        registration_binning = registration.get_optimal_registration_binning(sim1_0, sim2_0)
        scale_key, remaining_binning = msi_utils.get_res_level_from_binning_factors(msim1, registration_binning)
        sim1 = msi_utils.get_sim_from_msim(msim1, scale=scale_key)
        sim2 = msi_utils.get_sim_from_msim(msim2, scale=scale_key)
        if max(remaining_binning.values()) > 1:
            binned = []
            for sim in (sim1, sim2):
                sim_b = sim.coarsen(remaining_binning, boundary='trim').mean().astype(sim.dtype)
                sim_b.attrs.update(copy.deepcopy(sim.attrs))
                binned.append(sim_b)
            sim1, sim2 = binned

        return get_overlap_images(sim1, sim2, self.source_transform_key)

    def register_overlap(self, overlap1, overlap2, sims_pixel_space, params=None):
        """Run pairwise_reg_func directly on an already-extracted overlap crop (see
        select_pair_overlap()) - no msim-level resolution-selection or re-cropping, so a caller
        can reuse the same crop across repeated registration-parameter-only changes. Returns the
        physical-space transform, quality, and pairwise_reg_func's own raw result dict (feature
        points/matches, for a napari preview overlay).
        """
        params = params or {}
        _, pairwise_reg_func, pairwise_reg_func_kwargs = self.create_registration_method(overlap1, params=params)
        pairwise_reg_func_kwargs = dict(pairwise_reg_func_kwargs or {})

        fixed_data = overlap1.compute() if hasattr(overlap1, 'compute') else overlap1
        moving_data = overlap2.compute() if hasattr(overlap2, 'compute') else overlap2
        result = pairwise_reg_func(fixed_data, moving_data, **pairwise_reg_func_kwargs)

        try:
            transform = affine_from_intrinsic_affine(result['affine_matrix'], sims_pixel_space, self.source_transform_key)
        except NotImplementedError:
            transform = result['affine_matrix']

        return transform, result.get('quality'), result

    def create_fusion_method(self, fusion_method, sim0):
        if fusion_method is None:
            fusion_method = ''
        if 'compos' in fusion_method:
            fusion_method = None
            fuse_func = None
        elif 'exclus' in fusion_method:
            from muvis_align.fusion_methods.FusionMethodExclusive import FusionMethodExclusive
            fusion_method = FusionMethodExclusive(sim0, self.debug)
            fuse_func = fusion_method.fusion
        elif 'add' in fusion_method:
            from muvis_align.fusion_methods.FusionMethodAdditive import FusionMethodAdditive
            fusion_method = FusionMethodAdditive(sim0, self.debug)
            fuse_func = fusion_method.fusion
        else:
            fuse_func = fusion.simple_average_fusion

        self.fusion_method = fusion_method

        return fuse_func

    def register(self, register_msims=None, register_indices=None, params=None):
        pair_results = self.register_pairs(register_msims=register_msims, register_indices=register_indices, params=params)
        qualities = {key: metric[default_transform_key][default_quality_key]
                     for key, metric in pair_results['metrics']['pairs'].items()
                     if default_quality_key in metric[default_transform_key]}
        bboxes = {key: np.array(value.sel(t=0)).tolist() for key, value in nx.get_edge_attributes(self.pairs_graph, 'bbox').items()}
        self.save_pair_mappings(pair_results['pair_mappings'], qualities, bboxes)
        results = self.register_global(self.pair_msims, register_indices=register_indices, params=params)
        self.save_mappings(results['mappings'])
        self.save_metrics(results['metrics'])
        return results

    def register_pairs(self, register_msims=None, register_indices=None, params=None):
        if register_indices is None:
            if self.register_indices is not None:
                register_indices = self.register_indices
            else:
                register_indices = range(len(self.msims))

        operation = self.operation
        pairing = params.get('pairing',
                             params.get('registration', {}).get('pairing', '')).lower()
        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs',
                                              params.get('registration', {}).get('n_parallel_pairwise_regs'))
        if n_parallel_pairwise_regs is not None and n_parallel_pairwise_regs == '0':
            n_parallel_pairwise_regs = None

        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        is_stack = ('stack' in operation)

        reg_channel = params.get('channel', 0)
        if isinstance(reg_channel, int):
            reg_channel_index = reg_channel
            reg_channel = None
        else:
            reg_channel_index = None

        if register_msims is None:
            register_msims = self.register_msims

        if is_stack and not is_3d:
            # register in 2d; pairwise consecutive views - max-project every scale that has z
            # (map_msim_levels keeps register_msims a real, internally-consistent pyramid
            # throughout, rather than collapsing to a single level)
            def level_func(level_sim, scale_key):
                return si_utils.max_project_sim(level_sim, dim='z') if 'z' in level_sim.dims else level_sim
            register_msims = [map_msim_levels(msim, level_func) for msim in register_msims]
            pairs = [(index, index + 1) for index in range(len(register_msims) - 1)]
        elif 'ortho' in pairing or 'overla' in pairing:
            # position/size for pairing distance must match self.positions 1:1 (every source,
            # never a preprocessed/filtered register_msims subset) - self.msims (always the full,
            # untouched per-source pyramid) is exactly that, no sims needed for this metadata
            origins = np.array([get_sim_position_final(msi_utils.get_sim_from_msim(msim, scale='scale0'), position, get_center=True)
                                for msim, position in zip(self.msims, self.positions)])
            sizes = [get_sim_physical_size(get_msim_image0(msim)) for msim in self.msims]
            pairs, _ = get_pairs(origins, sizes, pairing)
            logging.info(f'#pairs: {len(pairs)}')
            #for pair in pairs:
            #    print(f'{self.file_labels[pair[0]]} - {self.file_labels[pair[1]]}')
        else:
            pairs = None

        # create_registration_method genuinely needs one concrete sim (feature-based/CPD methods
        # read real pixel data via cv2/skimage) - extracted on demand, once
        reg_method, pairwise_reg_func, pairwise_reg_func_kwargs = self.create_registration_method(
            msi_utils.get_sim_from_msim(register_msims[0], scale='scale0'), params=params)
        logging.info(f'Registration method: {reg_method}')
        logging.info('Registering...')
        # register_msims is a real multiscale, preprocessed pyramid per source (built by
        # preprocess()) - handing it to registration (instead of a single-level scale_factors=[]
        # wrap) lets compute_pairwise_registrations auto-select a good resolution per pair from
        # the real pyramid (no reg_res_level/registration_binning passed below)

        overlap_tolerance = 0

        # ******* start MVS registration functions

        # "c" in dims is read straight off each register_msim's own scale0 image DataArray
        # (get_msim_dims) - no sim needs to be built just to ask
        has_channel = ["c" in get_msim_dims(msim) for msim in register_msims]
        if has_channel[0]:
            if reg_channel is None:
                if reg_channel_index is None:
                    if any(has_channel):
                        raise Exception("Please choose a registration channel.")
                else:
                    reg_channel = get_msim_image0(register_msims[0]).coords["c"][reg_channel_index]

            msims_reg = [
                msi_utils.multiscale_sel_coords(msim, {"c": reg_channel})
                if has_channel[imsim]
                else msim
                for imsim, msim in enumerate(register_msims)
            ]
        else:
            msims_reg = register_msims
        
        # Normalize transforms to match image dimensions in each scale level
        # (handles mixed 3D/4D transforms in multiscale images)
        for msim in msims_reg:
            for scale_node in msim.ds.values():
                if 'source_metadata' in scale_node.ds.data_vars:
                    img_data = scale_node.ds['image']
                    if hasattr(img_data, 'data'):
                        img_data = img_data.data
                    # Get spatial dims from image shape
                    spatial_dims = [d for d in img_data.dims if d not in ('t', 'c')]
                    # Get current transform
                    current_transform = scale_node.ds.data_vars[self.source_transform_key]
                    transform_spatial_dims = [d for d in current_transform.coords['x_in'].values if d != '1']
                    # Adapt if mismatch
                    if len(transform_spatial_dims) != len(spatial_dims):
                        relevant_dim_names = spatial_dims + ['1']
                        adapted = current_transform.sel(x_in=relevant_dim_names, x_out=relevant_dim_names)
                        scale_node.ds[self.source_transform_key] = adapted

        try:
            with dask.config.set(scheduler='threads'):
                g_reg = mv_graph.build_view_adjacency_graph_from_msims(
                    msims_reg,
                    transform_key=self.source_transform_key,
                    pairs=pairs,
                    overlap_tolerance=overlap_tolerance,
                )

                g_reg_computed = compute_pairwise_registrations(
                    msims_reg,
                    g_reg,
                    transform_key=self.source_transform_key,
                    overlap_tolerance=overlap_tolerance,
                    pairwise_reg_func=pairwise_reg_func,
                    pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                    n_parallel_pairwise_regs=n_parallel_pairwise_regs,
                )

                # ******* end MVS registration functions

        except NotEnoughOverlapError:
            g_reg_computed = g_reg

        mappings = nx.get_edge_attributes(g_reg_computed, default_transform_key)
        mappings_dict = {(register_indices[indices[0]], register_indices[indices[1]]): mapping
                         for indices, mapping in mappings.items()}

        metrics = calc_pair_metrics(msims_reg, g_reg_computed, params.get('metrics', []), self.source_transform_key,
                                    reg_channel=reg_channel_index, n_parallel_pairs=n_parallel_pairwise_regs)

        self.pairs_graph = g_reg_computed
        self.pair_msims = msims_reg
        self.pairs = pairs
        self.metrics = metrics
        self.state = RegState.PAIRS_REG
        return {
            'pairs_graph': self.pairs_graph,
            'msims': msims_reg,
            'pairs': pairs,
            'pair_mappings': mappings_dict,
            'metrics': metrics
        }

    def register_global(self, pair_msims, register_indices=None, params=None,
                        pairs_graph=None):
        if register_indices is None:
            if self.register_indices is not None:
                register_indices = self.register_indices
            else:
                register_indices = range(len(pair_msims))

        if pairs_graph is not None:
            g_reg_computed = pairs_graph
        else:
            g_reg_computed = self.pairs_graph

        ndims = si_utils.get_ndim_from_sim(get_msim_image0(pair_msims[0]))

        groupwise_resolution_method = params.get('groupwise_resolution_method',
                                                 params.get('registration', {}).get('groupwise_resolution_method', 'global_optimization'))
        groupwise_resolution_kwargs = {}
        if groupwise_resolution_method == 'global_optimization':
           groupwise_resolution_kwargs['transform'] = params.get('transform_type',
                                                                 params.get('registration', {}).get('transform_type', 'affine'))
           # transform_type options include 'translation', 'rigid', 'affine', 'similarity'

        post_registration_quality_threshold = params.get('post_registration_quality_threshold',
                                                         params.get('registration', {}).get('post_registration_quality_threshold'))
        post_registration_do_quality_filter = (post_registration_quality_threshold is not None)

        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs',
                                              params.get('registration', {}).get('n_parallel_pairwise_regs'))
        if n_parallel_pairwise_regs is not None and n_parallel_pairwise_regs == '0':
            n_parallel_pairwise_regs = None

        plot_summary = self.mpl_ui

        # ******* start MVS registration functions

        if post_registration_do_quality_filter:
            # filter edges by quality
            g_reg_computed = mv_graph.filter_edges(
                g_reg_computed,
                threshold=post_registration_quality_threshold,
                weight_key="quality",
            )

        with dask.config.set(scheduler='threads'):
            transforms_dict, groupwise_resolution_info_dict = groupwise_resolution(
                g_reg_computed,
                method=groupwise_resolution_method,
                **groupwise_resolution_kwargs,
            )

        transforms = [
            transforms_dict[iview] for iview in sorted(g_reg_computed.nodes())
        ]

        for imsim, msim in enumerate(pair_msims):
            msi_utils.set_affine_transform(
                msim,
                transforms[imsim],
                transform_key=self.reg_transform_key,
                base_transform_key=self.source_transform_key,
            )

        if plot_summary:
            plot_info = _plot_registration_summaries(
                pair_msims,
                self.source_transform_key,
                self.reg_transform_key,
                g_reg_computed,
                groupwise_resolution_info_dict,
                show_plot=plot_summary,
            )
        else:
            plot_info = {}

        reg_result = {
            "params": transforms,
            "pairwise_registration": {
                "graph": g_reg_computed,
                "metrics": {
                    "qualities": nx.get_edge_attributes(
                        g_reg_computed, "quality"
                    )
                },
                "summary_plot": None if plot_summary is False
                else (
                    plot_info['fig_pair_reg'],
                    plot_info['ax_pair_reg']
                )
            },
            "groupwise_resolution": {
                "metrics": groupwise_resolution_info_dict,
                "summary_plot": None if plot_summary is False
                else (
                    plot_info['fig_group_res'],
                    plot_info['ax_group_res']
                )
            },
        }

        # ******* end MVS registration functions

        # copy transforms from the registration-stage msims onto self.msims (the persistent,
        # full per-source pyramid) - msim -> msim, writes the same affine onto every scale,
        # no sim round-trip needed
        for reg_msim, index in zip(pair_msims, register_indices):
            reg_transform = msi_utils.get_transform_from_msim(reg_msim, transform_key=self.reg_transform_key)
            msi_utils.set_affine_transform(self.msims[index], reg_transform, transform_key=self.reg_transform_key)

        # set missing transforms - sources that never took part in registration (e.g. filtered out)
        for msim in self.msims:
            if self.reg_transform_key not in get_msim_transform_keys(msim):
                msi_utils.set_affine_transform(
                    msim,
                    param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                    transform_key=self.reg_transform_key)

        mappings = reg_result['params']
        # re-index from subset of sims
        residual_error_dict = reg_result.get('groupwise_resolution', {}).get('metrics', {}).get('residuals', {})
        residual_error_dict = {(register_indices[key[0]], register_indices[key[1]]): value.item()
                               for key, value in residual_error_dict.items()}
        registration_qualities_dict = reg_result.get('pairwise_registration', {}).get('metrics', {}).get('qualities', {})
        registration_qualities_dict = {(register_indices[key[0]], register_indices[key[1]]): value
                                       for key, value in registration_qualities_dict.items()}

        # re-index from subset of sims
        mappings_dict = {index: mapping for index, mapping in zip(register_indices, mappings)}

        reg_channel = params.get('channel', 0)
        metrics = calc_global_metrics(pair_msims, self.source_transform_key, self.reg_transform_key,
                                      params.get('metrics', []), reg_channel=reg_channel, reg_results=reg_result,
                                      n_parallel_pairs=n_parallel_pairwise_regs)

        self.metrics = metrics
        self.state = RegState.GLOBAL_REG
        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'residual_errors': residual_error_dict,
                'registration_qualities': registration_qualities_dict,
                'metrics': metrics}

    def fuse(self, msims, fusion_method=None, output_spacing='mean', transform_key=None,
             dimension=None, output_filename=None,
             tile_size=None, ome_version=default_ome_zarr_version, extra_metadata=None,
             output_chunksize=None):
        """Fuse each source's own multiscale msim into one output msim (a real multiscale
        pyramid, not a single resolution) - msims in, msims out. A caller with only a concrete
        sim per source (an ad-hoc resolution with no corresponding real pyramid) wraps it into a
        trivial single-level msim first, via util.wrap_sims_as_msims - fuse() itself never takes
        or produces sims directly; extract a sim from the result on demand (msi_utils.
        get_sim_from_msim) wherever a downstream consumer (e.g. save_image()) needs one.

        output_chunksize, if given, is used as-is (a dict of chunk size per spatial dim, e.g.
        {'x':.., 'y':.., 'z':..} - a value larger than that dimension just yields one chunk
        covering it, dask clips it automatically). Otherwise fusion.fuse() defaults to the
        input's own on-disk chunk grid, which for a source chunked one z-slice at a time
        propagates that same z=1 chunking into every pyramid level of the fused output - each
        level then has as many chunks (and dask graph tasks) in z as there are z-slices, even
        once a level's XY extent has been downsampled to a handful of pixels.
        """
        if output_filename is not None:
            output_filename = self.output + output_filename

        # only .dtype is ever read from this below (data_size logging, create_fusion_method's
        # source_type) - the scale node's own 'image' DataArray already has it, no need to build
        # a full sim (with its attrs['transforms'] enrichment) just to read one attribute
        sim0 = get_msim_image0(msims[0])

        if extra_metadata is None:
            extra_metadata = self.extra_metadata

        channels = extra_metadata.get('channels', []) if isinstance(extra_metadata, dict) else []
        is_channel_overlay = ((dimension and dimension == 'c') or len(channels) > 1)

        if transform_key is None:
            transform_key = self.reg_transform_key

        if isinstance(self.source_metadata, dict):
            z_scale = self.source_metadata.get('scale', {}).get('z')
        elif isinstance(extra_metadata, dict):
            z_scale = extra_metadata.get('scale', {}).get('z')
        else:
            z_scale = None

        if z_scale is None:
            z_scale = extract_z_scale(self.positions, self.scales)

        z_positions = [position.get('z') for position in self.positions if 'z' in position]
        if len(set(z_positions)) > 1:
            msims = make_msims_3d(msims, z_scale=z_scale, positions=self.positions)

        output_stack_properties = calc_output_properties(msims, transform_key,
                                                         output_spacing_method=output_spacing, z_scale=z_scale)

        if self.verbose:
            logging.info(f'Output stack: {numpy_to_native(output_stack_properties)}')
        data_size = np.prod(list(output_stack_properties['shape'].values())) * sim0.dtype.itemsize
        logging.info(f'Fusing {print_hbytes(data_size)}')

        saving_zarr = False
        if is_channel_overlay:
            # convert to multichannel images - one channel per source, still a real multiscale
            # pyramid (combine_msims_as_channels stacks 'c' per level, not just at one resolution)
            def build_channel_result(msim):
                return fusion.fuse(
                    [msim],
                    transform_key=transform_key,
                    output_stack_properties=output_stack_properties,
                    output_chunksize=output_chunksize
                )

            # each fuse() call here only builds one source's own (lazy) dask graph against the
            # shared output_stack_properties - independent per source, so a thread pool spreads
            # that graph-construction work (real CPU cost for hundreds of sources) across every
            # available core instead of paying it out one source at a time. Joined synchronously
            # (results gathered in submission order below) before returning, same as
            # init_sources() - no async/background-worker handling needed on the caller's side.
            channel_results = [None] * len(msims)
            if len(msims) > 1:
                max_workers = min(default_preview_workers, len(msims))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(build_channel_result, msim): index
                              for index, msim in enumerate(msims)}
                    for future in as_completed(futures):
                        channel_results[futures[future]] = future.result()
            elif msims:
                channel_results[0] = build_channel_result(msims[0])
            fused_image = combine_msims_as_channels(channel_results, [channel['label'] for channel in channels])
        else:
            if fusion_method:
                logging.info(f'Fusion method: {fusion_method}')
            else:
                logging.info('Fusion method: [default method]')
            fuse_func = self.create_fusion_method(fusion_method, sim0)
            if fuse_func:
                saving_zarr = output_filename is not None
                if output_chunksize is None and saving_zarr and tile_size:
                    if not isinstance(tile_size, (list, tuple)):
                        tile_size = [tile_size] * 2
                    output_chunksize = xyz_to_dict(tile_size)
                    if 'z' in output_stack_properties['shape'] and 'z' not in output_chunksize:
                        # zarr export streams one z-slice at a time to keep peak memory low
                        output_chunksize['z'] = 1
                if saving_zarr:
                    if not output_filename.lower().endswith('.zarr'):
                        output_filename += zarr_extension
                    zarr_options = {'ome_zarr': saving_zarr, 'ngff_version': ome_version}
                else:
                    zarr_options = None
                with dask.config.set(scheduler='threads'):
                    fused_image = fusion.fuse(
                        msims,
                        fusion_func=fuse_func,
                        transform_key=transform_key,
                        output_stack_properties=output_stack_properties,
                        output_zarr_url=output_filename,
                        zarr_options=zarr_options,
                        output_chunksize=output_chunksize
                    )
            else:
                # 'compose' mode: no actual fusion, just return the per-source msims as-is
                fused_image = msims
        return fused_image, saving_zarr

    def save_pair_mappings(self, mappings, qualities, bboxes):
        pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
        output_mappings = {}
        for keys, mapping in mappings.items():
            label_key = json.dumps([self.filenames[keys[0]], self.filenames[keys[1]]])
            output_mappings[label_key] = {'mapping': np.array(mapping.sel(t=0)).tolist()}
            if keys in qualities:
                quality = quality_to_scalar(qualities[keys])
                output_mappings[label_key][default_quality_key] = float(quality)
            if keys in bboxes:
                output_mappings[label_key]['bbox'] = bboxes[keys]
        export_json(pair_mappings_filename, output_mappings)

    def save_mappings(self, mappings):
        mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
        output_mappings = {self.filenames[int(key)]: np.array(mapping.sel(t=0)).tolist()
                           for key, mapping in mappings.items()}
        write_transforms(mappings_filename, output_mappings, self.source_transform_key, self.reg_transform_key)

    def save_mappings_csv(self, mappings, normalise_orientation=False):
        data = []
        mappings_header = ['id',' filename', 'x_pixels', 'y_pixels', 'z_pixels', 'x', 'y', 'z', 'rotation']
        mappings_filename = self.output + self.output_params.get('mappings', default_mappings_tabular_name)
        for label, filename, msim, mapping, scale, position, rotation \
                in zip(self.file_labels, self.filenames, self.msims, mappings.values(), self.scales, self.positions, self.rotations):
            if not normalise_orientation:
                # rotation already in msim affine transform
                rotation = None
            position, rotation = get_data_mapping(msim,
                                                  transform_key=self.reg_transform_key,
                                                  transform=mapping,
                                                  translation0=position,
                                                  rotation=rotation)
            position_pixels = {dim: position[dim] / float(scale.get(dim, 1)) for dim in position.keys()}
            row = ([label] + [filename] + dict_to_xyz(position_pixels, add_zeros=True)
                   + dict_to_xyz(position, add_zeros=True) + [rotation])
            data.append(row)
        export_csv(mappings_filename, data, header=mappings_header)

    def save_metrics(self, metrics):
        metrics_filename = self.output + metrics_name
        output_metrics = {json.dumps([self.filenames[keys[0]], self.filenames[keys[1]]]):
                              {metric: float(value) for metric, value in metric_dict[self.reg_transform_key].items()}
                          for keys, metric_dict in metrics['pairs'].items() if metric_dict[self.reg_transform_key]}
        export_json(metrics_filename, output_metrics)

    def create_preview(self, output_filename=None, nom_msims=None, transform_key=None):
        output_params = self.params_general['output']
        preview_scale = output_params.get('preview_scale', 16)
        is_stack = ('stack' in self.operation)
        if isinstance(self.extra_metadata, dict):
            z_scale = self.extra_metadata.get('scale', {}).get('z')
        else:
            z_scale = None

        # select this preview resolution directly from self.msims (already built by init_data())
        # via pure msim slicing - no sim extraction, no resize, and (below) no wrapping back into
        # a msim just to satisfy fuse(): this stays msims the whole way through
        msims = select_msim_subpyramid_at_scale(self.msims, self.sources, preview_scale)
        if is_stack:
            msims = make_msims_3d(msims, z_scale, self.positions)

        if nom_msims is not None:
            # .sizes only - the scale node's own 'image' DataArray already has it
            preview_image0 = get_msim_image0(msims[0])
            nom_image0 = get_msim_image0(nom_msims[0])
            if preview_image0.sizes['x'] >= nom_image0.sizes['x']:
                logging.warning('Unable to generate scaled down preview due to lack of source pyramid sizes')
                return None

            if transform_key is not None and transform_key != self.source_transform_key:
                copy_transforms_to_msims(nom_msims, msims, transform_key)
        fusion_method = self.fusion_params.get('method', '')
        fused_msim, is_saved = self.fuse(msims, fusion_method=fusion_method, transform_key=transform_key,
                                         output_spacing='max', output_filename=output_filename)
        fused_sim = extract_sims_from_fused(fused_msim)
        if output_filename and (not is_saved or 'tif' in output_params.get('preview')):
            self.save(output_filename, fused_sim.squeeze(), transform_key=transform_key,
                      format=output_params.get('preview'), ome_version=output_params.get('ome_version'))
        return fused_sim

    def save(self, output_filename, data, format=zarr_extension, transform_key=None, translations0=None, channels=[],
             tile_size=None, compression=None, pyramid_downsample=2, npyramid_add=4, ome_version=default_ome_zarr_version):
        if output_filename is not None:
            output_filename = self.output + output_filename
        if isinstance(self.extra_metadata, dict) and self.extra_metadata:
            channels = self.extra_metadata.get('channels', [])
        save_image(output_filename, data, format,
                   transform_key=transform_key, channels=channels, translations0=translations0,
                   tile_size=tile_size, compression=compression,
                   pyramid_downsample=pyramid_downsample, npyramid_add=npyramid_add,
                   ome_version=ome_version,
                   verbose=self.verbose)

    def save_native_levels(self, output_filename, msim, position=None, channels=None, min_length=128,
                           compression=None, ome_version=default_ome_zarr_version):
        """Write msim's own native pyramid levels exactly as-is (no resampling) - see
        save_ome_multiscale_levels(). Used by convert, which never fuses/resamples a source.

        position: this source's own {'z': ..., ...} (self.positions[index]) - a source with no
        native 'z' dim (e.g. a single 2D tile) only carries its z height as this separate,
        external metadata, not in its own sim coords. Without promoting, the written file would
        silently drop that z position entirely instead of storing it as a size-1 'z' dim/
        translation - same promotion build_source_shape_sim()/make_msims_3d() already apply for
        shapes/fusion.
        """
        if output_filename is not None:
            output_filename = self.output + output_filename
        if channels is None:
            channels = self.extra_metadata.get('channels', []) if isinstance(self.extra_metadata, dict) else []

        if position is not None and 'z' not in get_msim_dims(msim):
            msim = make_msims_3d([msim], z_scale=self._msim_z_scale, positions=[position])[0]

        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        images = [msim[scale_key].ds['image'] for scale_key in scale_keys]
        dim_order = ''.join(images[0].dims)
        levels = [(image.data, si_utils.get_spacing_from_sim(image)) for image in images]
        translation = si_utils.get_origin_from_sim(images[0])

        save_ome_multiscale_levels(str(output_filename) + zarr_extension, levels, dim_order, channels,
                                   translation, min_length=min_length,
                                   compression=compression, ome_version=ome_version)

    def save_video(self, output, msims, fused_msim):
        logging.info('Creating transition video...')
        # rendering a video frame needs real pixel data (fusion.fuse, cv.resize, ...) - converted
        # to concrete sims here, once, rather than requiring the caller to have already done so
        sims = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims]
        fused_image = extract_sims_from_fused(fused_msim)
        pixel_size = [si_utils.get_spacing_from_sim(sims[0]).get(dim, 1) for dim in 'xy']
        params = self.params
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

    def get_metrics(self, metric_key=None, pair=None):
        metrics = self.metrics
        if pair is not None:
            if isinstance(pair, np.ndarray):
                pair = pair.tolist()
                pair = tuple(pair)
            metrics = metrics.get('pairs', {}).get(pair, {})
        else:
            if 'summary' in metrics:
                metrics = metrics['summary']
        reg_transform_key = self.reg_transform_key
        if reg_transform_key in metrics:
            transform_key = reg_transform_key
        elif default_transform_key in metrics:
            transform_key = default_transform_key
        elif metrics:
            transform_key = list(metrics)[-1]
        else:
            transform_key = None
        metrics = metrics.get(transform_key, {})
        if metric_key is not None:
            return metrics.get(metric_key)
        else:
            return metrics

    def output_exists(self, output_filename, output_format):
        if not output_format.startswith('.'):
            output_format = '.' + output_format
        output_filename = self.output + output_filename + output_format
        if output_format == zarr_extension:
            return (os.path.exists(os.path.join(output_filename, '.zattrs')) or
                    os.path.exists(os.path.join(output_filename, 'zarr.json')))
        else:
            return os.path.exists(output_filename)
