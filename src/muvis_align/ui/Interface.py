from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from enum import Enum, auto
from magicclass.ext.napari import ViewerWidget
from multiview_stitcher import spatial_image_utils as si_utils, param_utils
from napari.utils import progress
from napari.utils.notifications import show_warning
import networkx as nx
import numpy as np
import os.path
from xarray import DataTree
from qtpy.QtCore import QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QMessageBox

from muvis_align.constants import zarr_extension, default_transform_key, default_quality_key, \
    default_interactive_preview_scale, default_preview_workers
from muvis_align.file.project_yaml import read_params, get_template_params, write_params, update_params
from muvis_align.MVSRegistration import MVSRegistration, RegState
from muvis_align.image.util import get_sim_physical_size, get_sim_position_final, \
    create_image_shapes, create_overlap_shapes, build_source_shape_sim, \
    draw_keypoints_matches_napari, get_transforms, copy_transforms_to_msims, \
    make_msims_3d, metric_to_rgb, get_msim_level_data, get_contrast_limits, get_chunk_sizes, \
    get_msim_image0, wrap_sims_as_msims, extract_sims_from_fused, extract_sims_from_msims, \
    select_msim_subpyramid_at_scale
from muvis_align.file.resources import get_project_template
from muvis_align.logging import init_logging
from muvis_align.metrics import calc_msims_metrics
from muvis_align.Timer import Timer
from muvis_align.ui.NapariDaskProgress import NapariDaskProgress
from muvis_align.ui.NapariMVSProgress import NapariMVSProgress
from muvis_align.ui.NapariPreprocessProgress import NapariPreprocessProgress
from muvis_align.ui.ParamWidget import create_dict_of_lists, update_dict_value
from muvis_align.ui._utils import TemporarilyDisabledWidgets, VisibleActivityDock, catch_run_errors
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import print_dict_simple, set_dict_value, is_valid_value, \
    calculate_rigid_difference, operation_to_past_participle, eval_path, \
    resolve_to_project_dir, relativize_to_project_dir


class ViewMode(Enum):
    OVERVIEW = auto()
    PAIRS = auto()
    FEATURES = auto()
    FUSED = auto()


class Interface:
    def __init__(self, viewer, overview, enable_tabs=None, select_tab=None, is_tab_enabled=None,
                 enable_tab=None, enable_plugin_widget=None, verbose=False, initialize=True):
        self.viewer = viewer
        self.overview = overview
        self.enable_tabs = enable_tabs
        self.select_tab = select_tab
        self.is_tab_enabled = is_tab_enabled
        self.enable_tab = enable_tab
        self.enable_plugin_widget = enable_plugin_widget
        self.verbose = verbose
        self.raw_template = get_project_template()
        if not self.raw_template:
            raise FileNotFoundError('Project template not found')
        self.template = get_section_dict(self.raw_template, ['inputs', 'parameters', 'display_only', 'outputs'])
        self.param_widgets = {}
        self.params = {}
        self.pre_processing_performed = False
        self.metrics_methods = ['ncc', 'ssim', 'onmi']
        self.transform_key = 'source_metadata'
        self.need_source_reinit = False

        self.pair_metrics_timer = QTimer()
        self.pair_metrics_timer.setSingleShot(True)
        self.pair_metrics_timer.setInterval(1000)
        self.pair_metrics_timer.timeout.connect(self.update_pair_metrics)

        self.reg = MVSRegistration()
        if initialize:
            self.reset()

    def reset(self):
        self.source_metadata = {}
        self.extra_metadata = {}
        self.output_channels = []
        self.view_mode = None
        self.selected_shape_index = None
        self._preview_overlap_cache = None
        self._view_msims = None
        self.reg.reset()
        self._clear_napari_view(self.overview)
        self._clear_napari_view(self.viewer)
        if self.enable_tabs:
            self.enable_tabs(False, 2)
        if self.select_tab:
            self.select_tab(1)

    def get_all_widgets(self):
        # excludes widgets on a currently disabled tab - their .enabled always reads False
        # (inherited from the disabled tab page), so snapshotting and restoring it as an explicit
        # per-widget state (see modify_pair_registration) would leave them disabled even after
        # their tab is enabled again
        return {name: param_widget.widget for name, param_widget in self.param_widgets.items()
               if self.is_tab_enabled is None or self.is_tab_enabled(name.split('.', 1)[0])}

    def get_function(self, function_label):
        if hasattr(self, function_label):
            return eval(f'self.{function_label}')
        else:
            return None

    def tab_changed(self, tab_label):
        if tab_label != 'registration' and self.view_mode == ViewMode.FEATURES:
            self._clear_napari_view(self.viewer)
            self.view_mode = None
        self.pair_metrics_timer.stop()

    def project_path(self, path):
        self.reset()
        self.params_path = path
        self.params = get_template_params(self.template)
        if os.path.exists(path):
            self.params = update_params(self.params, read_params(path))
            self.update_widgets()
        else:
            self.write_params()
        self.update_input_output_path()

    def get_project_dir(self):
        # input/output path params are stored relative to this directory, so a project stays
        # portable when the project file and its data are moved or shared together
        params_path = getattr(self, 'params_path', None)
        return os.path.dirname(os.path.abspath(params_path)) if params_path else None

    def update_widgets(self):
        for param_name, param_widget in self.param_widgets.items():
            # input/output path widgets are handled separately by update_input_output_path(),
            # which resolves them relative to the project directory before display
            if param_name not in ('input_output.input_path', 'input_output.output_path'):
                keys = param_name.split('.')
                value = self.params.get(keys[0], {}).get(keys[1])
                if value is not None:
                    param_widget.set_value(value)

    def write_params(self):
        write_params(self.params_path, self.params)

    def change_param(self, param_name, value):
        keys = param_name.split('.')
        if keys[0] not in self.params:
            self.params[keys[0]] = {}
        if isinstance(value, str):
            value = value.replace('\\', '/')
            if param_name in ('input_output.input_path', 'input_output.output_path'):
                # the file dialog (and the FileEdit widget itself) always reports an
                # absolute path - convert it back to relative-to-project-dir before storing,
                # so the project file keeps portable relative paths
                value = relativize_to_project_dir(value, self.get_project_dir())
        self.params[keys[0]][keys[1]] = value
        self.write_params()

    def update_input_output_path(self):
        # display the path exactly as stored (relative-to-project-dir when the project file
        # keeps it relative) - FileEdit.set_value() would force it absolute, so the inner
        # line edit's text is set directly instead, bypassing that conversion
        params = self.params['input_output']
        widget = self.param_widgets.get('input_output.input_path')
        input_path = params.get('input_path', '')
        if isinstance(eval_path(input_path), str):
            self._set_path_widget_text(widget, input_path)
        widget = self.param_widgets.get('input_output.output_path')
        output_path = params.get('output_path', '')
        if isinstance(eval_path(output_path), str):
            self._set_path_widget_text(widget, output_path)
        resolved_output_path = resolve_to_project_dir(output_path, self.get_project_dir())
        init_logging(log_filename=os.path.join(resolved_output_path, 'muvis-align.log'), verbose=self.verbose)

    def _set_path_widget_text(self, param_widget, value):
        line_edit = getattr(param_widget.widget, 'line_edit', None)
        if line_edit is not None:
            line_edit.value = value
        else:
            param_widget.set_value(value)

    def input_path(self, value):
        self.need_source_reinit = True

    def source_position_z(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'z'], value)
            self.need_source_reinit = True

    def source_position_y(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'y'], value)
            self.need_source_reinit = True

    def source_position_x(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'x'], value)
            self.need_source_reinit = True

    def source_scale_z(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'z'], value)
            self.need_source_reinit = True

    def source_scale_y(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'y'], value)
            self.need_source_reinit = True

    def source_scale_x(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'x'], value)
            self.need_source_reinit = True

    def source_rotation(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['rotation'], value)
            self.need_source_reinit = True

    def registration_dimension(self, value):
        # Force reinitialization of extra metadata / channels when registration dimension changes
        self.extra_metadata.pop('channels', None)

    def channels_table(self, value):
        old_value = self.param_widgets.get('input_output.channels_table').get_value()
        channels_dict = update_dict_value(old_value, value)
        channels = [{'label': label} for label in channels_dict['label']]
        for channeli, channel in enumerate(channels):
            if channeli < len(channels_dict['color']):
                color = channels_dict['color'][channeli]
                try:
                    if color and isinstance(color, str):
                        channel['color'] = tuple(eval(color))
                except Exception:
                    pass
        self.extra_metadata['channels'] = channels

    def input_output_process(self):
        params = self.params['input_output']
        project_dir = self.get_project_dir()
        output = resolve_to_project_dir(str(params['output_path']), project_dir)
        if not self.reg.is_initialised() or self.need_source_reinit:
            self.need_source_reinit = False
            if not output.endswith('/'):
                output += '/'
            input_path = resolve_to_project_dir(params['input_path'], project_dir)
            ok = self.reg.init(input_path=eval_path(input_path),
                               output_path=output,
                               overwrite=params['overwrite'],
                               verbose=self.verbose)
            if ok:
                # init_progress(), right below, always ends by drawing the view (with
                # whatever transform its own registration-state check picks) - drawing here
                # too would just draw once with the not-yet-registered transform, immediately
                # thrown away by that second, correct draw.
                ok = self.update_metadata_source(skip_view_update=True)
                if ok:
                    self.populate_image_selection()
                    self.init_progress()
            if not ok:
                show_warning('Invalid input or output')
                self.reg.state = RegState.UNINIT
        elif self.reg.is_global_registered():
            self.update_registered(view_transform_key=self.reg.reg_transform_key)
        elif self.reg.is_pairs_registered():
            self.update_registered(view_transform_key=self.reg.source_transform_key)
        else:
            self.update_metadata_source()

    def init_progress(self):
        output_filename = operation_to_past_participle(self.params['registration']['operation'])
        self.reg.init_progress(output_filename, zarr_extension)
        if self.reg.is_pairs_registered() and self.reg.register_msims is None:
            # reg.init_progress() can load a previously-saved pair registration from disk (a
            # reloaded project) - in that branch it sets pair_msims straight from
            # self.reg.msims, the raw full pyramid, bypassing the scale-reduction a live
            # pair-registration run always applies first (preprocess()'s
            # select_msim_subpyramid_at_scale). Left unprocessed, a later global
            # registration's metrics computation can auto-select the full-resolution level
            # and run out of memory. Run the same preprocessing now and re-point pair_msims
            # at its result, matching what a fresh run would have produced.
            self.run_pre_processing()
            self.reg.pair_msims = self.reg.register_msims
        if self.reg.is_fused():
            copy_transforms_to_msims(self.reg.msims, self.view_msims, self.reg.reg_transform_key)
            self.preview_fusion()
            self.enable_tabs(True, 4)
            self.select_tab(4)
        elif self.reg.is_global_registered():
            copy_transforms_to_msims(self.reg.msims, self.view_msims, self.reg.reg_transform_key)
            self.update_registered(view_transform_key=self.reg.reg_transform_key)
            self.enable_tabs(True, 4)
            self.select_tab(4)
        elif self.reg.is_pairs_registered():
            self.update_registered(view_transform_key=self.reg.source_transform_key)
            self.enable_tabs(True, 3)
            self.select_tab(3)
        else:
            # No prior registration to view with a specific transform - this is the one
            # draw update_metadata_source()'s own (skipped, see input_output_process())
            # would otherwise have done for a brand-new project. No pre-processing has run
            # yet, so only shapes are shown - see update_views()'s show_images param.
            self.update_views(show_images=False)
            self.enable_tabs(True, 2)

    def update_metadata_source(self, skip_view_update=False):
        if not self.reg.is_pairs_registered():
            try:
                with NapariPreprocessProgress(progress_class=progress,
                                              desc='Initialising sources',
                                              bar_format=" ",
                                              min_duration=0.1) as progress_factory, \
                     TemporarilyDisabledWidgets(self.enable_plugin_widget), \
                     VisibleActivityDock(self.viewer):
                    self.reg.init_data(
                        source_metadata=self.source_metadata,
                        progress_factory=progress_factory,
                    )
            except ValueError as e:
                show_warning('Unable to read source data\n' + str(e))
                logging.exception('Unable to read source data')
                return False

            # the expensive part - actually building each source's multiscale msim (and, for a
            # multi-z stack, wrapping the whole set into a 3D volume) - is deferred to the lazy
            # view_msims property below, only forced once something genuinely needs pixel-shaped
            # data (the fused image preview, once shown after pre-processing). The z-index
            # remapping itself is cheap (plain dict/set bookkeeping) and stays eager here so
            # self.reg.positions is immediately correct for shapes, drawn instantly either way.
            z_positions = sorted(set([position.get('z', 0) for position in self.reg.positions]))
            if len(z_positions) > 1:
                for position in self.reg.positions:
                    position['z'] = z_positions.index(position.get('z', 0))
            self._view_msims = None
        # this function only ever runs pre-registration (both call sites guard on
        # `not self.reg.is_pairs_registered()`), so self.reg.source_transform_key is the only
        # transform key that can exist yet - no need to force self.reg.msims just to read it
        # off a msim that would say the same thing
        coord_systems = [self.reg.source_transform_key]
        self.populate_channels()
        self.populate_coordinate_systems(coord_systems)
        if self.update_output_channels():
            self.populate_channels_table()
        if self.reg.is_initialised():
            # populate_metadata_table() never reads its first arg when transform_keys is None
            # (it reads self.reg.positions/scales directly) - passing None instead of
            # self.reg.msims avoids forcing the expensive msim build just for this call
            self.populate_metadata_table(None)
            self.check_3d_view()
            if not skip_view_update:
                self.update_views(show_images=False)

        return True

    @property
    def view_msims(self):
        # per-source msim for the napari image data layer - built lazily, only once something
        # (the fused image preview) actually needs it; shapes no longer depend on this at all
        # (see _create_napari_shapes(), which builds its own cheap per-source sims directly)
        if self._view_msims is None:
            self._view_msims = self._build_view_msims()
            if len(set(position.get('z', 0) for position in self.reg.positions)) > 1:
                self._view_msims = make_msims_3d(self._view_msims, positions=self.reg.positions)
        return self._view_msims

    @view_msims.setter
    def view_msims(self, value):
        self._view_msims = value

    def _build_view_msims(self):
        # per-source msim for the napari image data layer: a source with a native multi-
        # resolution pyramid is used as-is; a single-resolution source is downscaled by one
        # constant factor when its largest spatial dimension exceeds 1000px
        view_msims = []
        for source, msim in zip(self.reg.sources, self.reg.msims):
            if len(source.shapes) == 1:
                image0 = get_msim_image0(msim)
                spatial_dims = si_utils.get_spatial_dims_from_sim(image0)
                largest_dim = max(image0.sizes[dim] for dim in spatial_dims)
                if largest_dim > 1000:
                    scale_factor = largest_dim / 1000
                    sim = extract_sims_from_msims(
                        [msim], [source], self.reg.source_transform_key, target_scale=scale_factor
                    )[0]
                    msim = wrap_sims_as_msims([sim])[0]
            view_msims.append(msim)
        return view_msims

    @catch_run_errors
    def run_pre_processing(self):
        params_features = self.params['pre_processing']
        with NapariPreprocessProgress(progress_class=progress,
                                      desc='Pre-processing',
                                      bar_format=" ",
                                      min_duration=0.1) as progress_factory, \
             TemporarilyDisabledWidgets(self.enable_plugin_widget), \
             VisibleActivityDock(self.viewer), \
             Timer('pre_processing_process', verbose=self._timing_verbose()):
            # self.reg.msims is built lazily (see MVSRegistration.msims) - building it here
            # explicitly, through ensure_msims(), gives that per-source construction its own
            # progress reporting instead of it happening silently (no progress feedback) as a
            # side effect of evaluating `self.reg.msims` as a plain argument below
            with Timer('run_pre_processing: build msims (load image data)', verbose=self._timing_verbose()):
                msims = self.reg.ensure_msims(progress_factory=progress_factory)
            with Timer('run_pre_processing: preprocess', verbose=self._timing_verbose()):
                _, _, modified = self.reg.preprocess(msims,
                                                     progress_factory=progress_factory,
                                                     **params_features)
        self.pre_processing_performed = modified
        return True

    def pre_processing_process(self):
        if not self.run_pre_processing():
            return
        self.update_views(show_preprocessed=True)
        self.enable_tabs(True, 3)
        self.enable_modify_pair_registration(False)
        self.select_tab(3)

    def populate_channels(self):
        channel_labels = list({channel.get('label', '') for source in self.reg.sources for channel in source.get_channels()})
        choices = {channel: channel for channel in channel_labels}
        param_widget = self.param_widgets.get('registration.channel')
        param_widget.set_choices(choices)

    def populate_coordinate_systems(self, coord_systems):
        choices = {coord_system: coord_system.replace('_', ' ').capitalize() for coord_system in coord_systems}
        param_widget = self.param_widgets.get('input_output.coordinate_system')
        param_widget.set_choices(choices)

    def coordinate_system(self, transform_key):
        self.transform_key = transform_key
        if self.reg.is_initialised():
            self.populate_metadata_table(self.reg.msims, [transform_key])

    def populate_metadata_table(self, sims, transform_keys=None):
        # https://pyapp-kit.github.io/magicgui/api/widgets/Table/
        # https://pyapp-kit.github.io/magicgui/generated_examples/demo_widgets/table/
        table_widget = self.param_widgets.get('input_output.metadata_table')
        properties = ['position', 'size']
        if transform_keys is None:
            positions = self.reg.positions
            scales = self.reg.scales
        else:
            positions = [get_sim_position_final(sim, transform_keys=transform_keys) for sim in sims]
            scales = [get_sim_physical_size(sim) for sim in sims]
        data = [[print_dict_simple(position),
                 print_dict_simple(scale)]
                for position, scale in zip(positions, scales)]
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((data, self.reg.file_labels, properties))
        table_widget.set_table_column_resize_mode()

    def update_output_channels(self):
        if not self.extra_metadata.get('channels'):
            # get channels from source
            source0 = self.reg.sources[0]
            channels = source0.get_channels()

            dimension = self.params['input_output']['registration_dimension']
            while dimension.lower() == 'c' and len(channels) < len(self.reg.sources):
                channel = {'label': f'channel {len(channels)}'}
                channels.append(channel)

            self.extra_metadata['channels'] = channels

            # convert to list dict
            data = [[channel.get('label', f'channel {index}'), channel.get('color', (1, 1, 1))]
                     for index, channel in enumerate(channels)]
            self.output_channels = create_dict_of_lists(data, ['label', 'color'])
            return True

        return False

    def populate_channels_table(self):
        param_widget = self.param_widgets.get('input_output.channels_table')
        param_widget.set_value(self.output_channels)

    def populate_image_selection(self):
        labels = self.reg.file_labels
        widget1 = self.param_widgets.get('registration.reg_preview_image1')
        widget1.set_value(labels[0], choices=labels)

        widget2 = self.param_widgets.get('registration.reg_preview_image2')
        index = 1 if len(labels) > 1 else 0
        widget2.set_value(labels[index], choices=labels)

    def get_best_transform_key(self):
        if not self.reg.is_pairs_registered():
            # only self.reg.source_transform_key can exist yet - avoid forcing the expensive
            # msim build (get_transforms() needs self.reg.msims) just to learn that
            return self.reg.source_transform_key
        transforms = get_transforms(self.reg.msims)
        if self.reg.reg_transform_key in transforms:
            transform_key = self.reg.reg_transform_key
        elif default_transform_key in transforms:
            transform_key = default_transform_key
        elif self.reg.source_transform_key in transforms:
            transform_key = self.reg.source_transform_key
        else:
            transform_key = None
        return transform_key

    def check_3d_view(self):
        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        ndisplay = 3 if is_3d else 2
        self.viewer.dims.ndisplay = ndisplay
        #self.overview._qtwidget._viewer_model.dims.ndisplay = ndisplay

    def _timing_verbose(self):
        # a few view-update methods below are unit-tested against a bare Interface with no
        # self.reg at all (e.g. _napari_view_add_fused_data) - fall back to no timing logging
        # rather than requiring self.reg just to gate these diagnostic Timer() calls
        return getattr(getattr(self, 'reg', None), 'logging_time', False)

    def update_views(self, transform_key=None, show_preprocessed=False, show_images=True):
        if transform_key is None:
            transform_key = self.get_best_transform_key()

        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        is_multi_z_shapes = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
        force_2d = is_multi_z_shapes and not is_3d
        with Timer('update_views: create shapes', verbose=self._timing_verbose()):
            shapes, refs, labels, face_colors = self._create_napari_shapes(transform_key, force_2d=force_2d)

        self._clear_napari_view(self.viewer)
        # before pre-processing has run, only shapes are ever shown (show_images=False) - the
        # fused image preview always needs every source's real msim built (see
        # _create_napari_data()), so showing it only once pre-processing/registration has
        # actually happened is also what keeps that cost from ever blocking initial project load
        if show_images:
            with Timer('update_views: create fused data', verbose=self._timing_verbose()):
                data = self._create_napari_data(transform_key, show_preprocessed=show_preprocessed)
            if data is not None:
                with Timer('update_views: add fused data to viewer', verbose=self._timing_verbose()):
                    # cheap=True: this is the general overview, not the accurate fusion-tab
                    # preview (preview_fusion()) or the real exported result (fusion_process())
                    # - a naive contrast guess is fine here, see _napari_view_add_fused_data()
                    self._napari_view_add_fused_data(self.viewer, data, f'{self.reg.fileset_label} data',
                                                     cheap=True)
        with Timer('update_views: add shapes to viewer', verbose=self._timing_verbose()):
            self._update_view_add_shapes(self.viewer, shapes, refs, labels, face_colors, f'{self.reg.fileset_label} shapes')

        with Timer('update_views: refresh overview shapes', verbose=self._timing_verbose()):
            self._refresh_overview_shapes(transform_key, shapes, refs, labels, face_colors, is_3d=is_3d)
        self.view_mode = ViewMode.OVERVIEW

    def _refresh_overview_shapes(self, transform_key, shapes=None, refs=None, labels=None,
                                 face_colors=None, is_3d=None):
        # the overview widget always shows a flattened, top-down layout, independent of
        # whatever the main viewer currently shows (a fused image, per-tile overview, or
        # nothing loaded yet) - a genuinely 3D shape set (drawn as oriented 3D boxes for the
        # main viewer) needs recomputing with force_2d=True for it instead of being reused as-is
        if is_3d is None:
            is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        if shapes is None or is_3d:
            shapes, refs, labels, face_colors = self._create_napari_shapes(transform_key, force_2d=True)
        self._clear_napari_view(self.overview)
        self._update_view_add_shapes(self.overview, shapes, refs, labels, face_colors, f'{self.reg.fileset_label} shapes')

    def _clear_napari_view(self, viewer):
        # Avoid emitting an empty LayerList.clear() event.  Under xpra/Xvfb,
        # that event can leave napari's VisPy canvas with broken blending for
        # subsequently created Shapes and Points layers.
        if viewer is not None and len(viewer.layers) > 0:
            viewer.layers.clear()

    def _create_napari_shapes(self, transform_key, force_2d=False):
        if transform_key == self.reg.source_transform_key:
            # not yet registered (or explicitly asking for original positions) - build cheap,
            # single-level sims straight from the already-resolved per-source geometry, never
            # touching self.reg.msims/self.view_msims (the expensive multiscale msim build) -
            # any other transform_key (e.g. 'registered') means registration has already run,
            # so self.view_msims is legitimately available and carries that transform already
            #
            # promote_z mirrors the make_msims_3d() promotion self.view_msims itself gets: when
            # output_order has no native 'z' (every source is individually 2D) but different
            # sources sit at different z heights, each source's own z position must still become
            # a real 'z' dim on its sim - otherwise it's silently dropped instead of drawn at its
            # actual z.
            promote_z = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
            with Timer(f'_create_napari_shapes: build {len(self.reg.sources)} source shape sims',
                      verbose=self._timing_verbose()):
                msims = [
                    build_source_shape_sim(source, self.reg._msim_output_order, translation, transform,
                                           transform_key, z_scale=self.reg._msim_z_scale, promote_z=promote_z)
                    for source, translation, transform in
                    zip(self.reg.sources, self.reg.positions, self.reg._msim_transforms)
                ]
        else:
            with Timer('_create_napari_shapes: get view_msims', verbose=self._timing_verbose()):
                msims = self.view_msims

        with Timer(f'_create_napari_shapes: create_image_shapes ({len(msims)} images)', verbose=self._timing_verbose()):
            shapes = create_image_shapes(msims, transform_key=transform_key, force_2d=force_2d)
        refs = [str(index) for index in range(len(msims))]
        labels = list(self.reg.file_labels)
        face_colors = [(1, 1, 1) for _ in range(len(msims))]

        # Once pairwise registration has run, restrict overlap boxes to pairs that were
        # actually registered (real edges in pairs_graph) - a geometric intersection that
        # only appears after alignment, between images that were never paired during
        # registration in the first place, has no quality score behind it and shouldn't be
        # drawn. Before registration there's no graph yet, so fall back to every
        # geometrically-overlapping pair (create_overlap_shapes' own default).
        overlap_pairs = list(self.reg.pairs_graph.edges()) if self.reg.is_pairs_registered() else None
        with Timer(f'_create_napari_shapes: create_overlap_shapes ({len(msims)} images,'
                  f' {len(overlap_pairs) if overlap_pairs is not None else "all"} pairs)', verbose=self._timing_verbose()):
            shapes2, pairs = create_overlap_shapes(msims, transform_key=transform_key, pairs=overlap_pairs,
                                                   force_2d=force_2d)
        shapes.extend(shapes2)
        refs += [f'{index1} {index2}' for index1, index2 in pairs]
        labels += ['' for _ in pairs]
        face_colors += [np.array(metric_to_rgb(self.reg.get_metrics(default_quality_key, pair))) for pair in pairs]
        return shapes, refs, labels, face_colors

    def _create_napari_data(self, transform_key, fusion_method='additive', show_preprocessed=False):
        if show_preprocessed:
            # copy to avoid transform changes below leaking into the stored register_msims
            with Timer(f'_create_napari_data: copy {len(self.reg.register_msims)} register_msims',
                      verbose=self._timing_verbose()):
                msims = [msim.copy(deep=True) for msim in self.reg.register_msims]
        else:
            # view_msims (unlike register_msims) is never scale-reduced - every source's full
            # native pyramid, un-preprocessed. Fusing that at native/scale0 resolution just to
            # draw an on-screen overview forces building (and, for get_contrast_limits(), running)
            # the graph for the largest, most expensive levels of the combined output, even though
            # only its coarsest handful of pixels is ever actually shown until the user zooms in.
            # Reduce to the same kind of coarse sub-pyramid MVSRegistration.create_preview() already
            # uses for its own (exported) preview, rather than fusing every level of every source.
            preview_scale = self.params['input_output'].get('preview_scale', default_interactive_preview_scale)
            with Timer('_create_napari_data: build view_msims', verbose=self._timing_verbose()):
                view_msims = self.view_msims
            with Timer(f'_create_napari_data: select_msim_subpyramid_at_scale ({len(view_msims)} images)',
                      verbose=self._timing_verbose()):
                msims = select_msim_subpyramid_at_scale(view_msims, self.reg.sources, preview_scale)
        with Timer('_create_napari_data: copy_transforms_to_msims', verbose=self._timing_verbose()):
            copy_transforms_to_msims(self.reg.msims, msims, transform_key)
        # A source chunked one z-slice at a time on disk would otherwise propagate that
        # z=1 chunking into every pyramid level of the preview (each level then has as many
        # dask tasks in z as there are z-slices, even once XY has been downsampled to a
        # handful of pixels), ballooning dask graph-construction time. But a single chunk
        # spanning the whole z range is just as bad from the other direction: napari can only
        # compute a chunk as a whole, so viewing one z-slice forces fusing every slice in its
        # chunk. get_chunk_sizes keeps x/y generous (shown in full for any view) and derives a
        # small z chunk from the byte budget instead, since z is what napari slices through.
        image0 = get_msim_image0(msims[0])
        spatial_dims = si_utils.get_spatial_dims_from_sim(image0)
        # MVSRegistration.fuse() (below) promotes msims to 3D internally (make_msims_3d) whenever
        # sources sit at more than one distinct z position, regardless of whether msims already
        # has a 'z' dim here - output_chunksize must already account for that dim in that case,
        # since fuse() computes its own output_stack_properties from the (by-then 3D) msims, not
        # from whatever's passed in here as output_chunksize
        z_positions = [position.get('z') for position in self.reg.positions if 'z' in position]
        if len(set(z_positions)) > 1 and 'z' not in spatial_dims:
            spatial_dims = ['z'] + spatial_dims
        output_chunksize = get_chunk_sizes(image0.dtype, spatial_dims)
        with Timer(f'_create_napari_data: fuse ({len(msims)} images)', verbose=self._timing_verbose()):
            fused_msim, _ = self.reg.fuse(msims,
                                          transform_key=transform_key,
                                          fusion_method=fusion_method,
                                          dimension=self.params['input_output']['registration_dimension'],
                                          extra_metadata=self.extra_metadata,
                                          output_chunksize=output_chunksize)
        return fused_msim

    def _update_view_add_shapes(self, viewer, shapes, refs, labels, face_colors, layer_name):
        # is_3d/is_multi_z_shapes read cheap per-source metadata (self.reg.sources/positions)
        # rather than self.view_msims, so drawing shapes never forces the expensive msim build
        # this is otherwise deferred to the fused image preview / pre-processing
        bb_supported = True
        if isinstance(viewer, ViewerWidget):
            viewer = viewer._qtwidget._viewer_model
            bb_supported = False
        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        is_multi_z_shapes = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
        force_2d = not bb_supported or (is_multi_z_shapes and not is_3d)
        # a shape actually carries a 'z' coordinate whenever the source is natively a z-stack
        # (is_3d) or build_source_shape_sim()/make_msims_3d() promoted it to one (is_multi_z_shapes)
        do_3d = ((is_3d or is_multi_z_shapes) and not force_2d)

        if len(shapes) > 0:
            # Depth-tested 'translucent' made overlap boxes lose to the opaque fused image
            # volume they sit inside (worse, not better) - translucent_no_depth avoids that
            # for both 2D and 3D, so shapes always draw regardless of what else is there.
            blending = 'translucent_no_depth'
            edge_color = 'cyan'
            if do_3d:
                # A 'polygon' shape can only render one flat face, not a whole non-planar box,
                # and napari-bbox 0.1.1 (which drew real 3D boxes) is incompatible with current
                # napari - so each box becomes its own 6 flat quad faces (for an opaque,
                # quality-colored box) plus one edge-only 'path' outlining every edge (for a
                # crisp wireframe), combined into one Shapes layer via per-shape lists. Corner
                # order is _minimal_bb_vertices': 0-3 one face, 4-7 the opposite face, in the
                # same winding, so corner i and i+4 are the vertical edges between them.
                box_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
                edge_path = [0, 1, 2, 3, 0, 4, 7, 3, 2, 6, 7, 4, 5, 6, 2, 1, 5]
                # Matches _minimal_bb_vertices' own corner_bits convention (bottom face 0-3,
                # top face 4-7, same x/y winding), applied here to plain axis-aligned min/max
                # instead of an oriented frame.
                corner_bits = np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                ])

                face_shapes, face_only_colors, face_refs, face_labels = [], [], [], []
                for shape, ref, color in zip(shapes, refs, face_colors):
                    corners = np.asarray(shape)
                    # napari doesn't render a 3D polygon's face fill unless that face's own
                    # plane is axis-orthogonal (napari/napari#6860) - a genuinely oriented/
                    # rotated box (as global registration can produce) gets no face fill at
                    # all except for whichever face happens to be axis-aligned. Build the
                    # fill from this box's axis-aligned bounding box instead (always
                    # axis-orthogonal by construction); the wireframe below keeps using the
                    # true oriented corners, which render fine as edges regardless of angle.
                    mins, maxs = corners.min(axis=0), corners.max(axis=0)
                    aa_corners = mins + corner_bits * (maxs - mins)
                    centroid = aa_corners.mean(axis=0)
                    for face in box_faces:
                        quad = aa_corners[face]
                        # box_faces' index order alone doesn't guarantee consistent outward
                        # winding (e.g. face [0,1,2,3] winds inward, [4,5,6,7] outward, even
                        # for a plain axis-aligned box) - flip it if the face's own normal
                        # doesn't point away from the box's center, or napari would
                        # backface-cull roughly half of every box's faces.
                        normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
                        if np.dot(normal, quad.mean(axis=0) - centroid) < 0:
                            quad = quad[::-1]
                        face_shapes.append(quad)
                    face_only_colors += [color] * len(box_faces)
                    face_refs += [ref] * len(box_faces)
                    face_labels += [''] * len(box_faces)
                wire_shapes = [np.asarray(shape)[edge_path] for shape in shapes]

                # Napari renders 3D paths as tubes whose width is measured in world
                # coordinates, not screen pixels. Scale the tube radius to the scene so it
                # remains visible for large physical units.
                vertices = np.concatenate([np.asarray(shape) for shape in shapes])
                wire_width = np.ptp(vertices, axis=0).max() * 0.005

                shape_data = face_shapes + wire_shapes
                shape_type = ['polygon'] * len(face_shapes) + ['path'] * len(wire_shapes)
                # face_color/edge_color entries must all be the same length (napari can't
                # build one color array from mixed 3- and 4-tuples, and silently falls back
                # to plain white for the whole layer instead) - a 'path' has no face to color
                # and its edge is drawn regardless of face_color, so (0, 0, 0) here is just a
                # length-matched placeholder, not a meaningful value.
                face_color = face_only_colors + [(0, 0, 0)] * len(wire_shapes)
                edge_color = [(0, 0, 0)] * len(face_shapes) + [(0, 1, 1)] * len(wire_shapes)
                edge_width = [0] * len(face_shapes) + [wire_width] * len(wire_shapes)
                refs = face_refs + refs
                labels = face_labels + labels
            else:
                shape_data = np.asarray(shapes)
                shape_type = 'polygon'
                edge_width = 0.1
                face_color = face_colors

            # the overview widget draws the same shapes much smaller on screen than the main
            # viewer - napari's default text size (12) reads oversized there, so halve it
            text_size = 6 if not bb_supported else 12
            text = {'string': '{labels}', 'size': text_size}
            features = {'refs': refs, 'labels': labels}
            viewer.add_shapes(shape_data, name=layer_name, shape_type=shape_type, text=text, features=features,
                              face_color=face_color, opacity=0.5, edge_width=edge_width, edge_color=edge_color,
                              blending=blending)

            # layer = viewer.add_shapes(shapes, name=layer_name, text=text, features=features, opacity=0.5,
            #                           face_color=face_colors)
            # @viewer.mouse_move_callbacks.append
            # def on_mouse_move(viewer, event):
            #     self.selected_shape_index = layer._value[0]
            #
            # @viewer.mouse_drag_callbacks.append
            # def on_mouse_drag(viewer, event):
            #     if event.type == "mouse_press" and event.button == 1:
            #         if viewer.layers.selection.active == layer and self.selected_shape_index is not None:
            #             self.on_selection_change(refs[self.selected_shape_index])
            #     yield

    def _napari_view_add_fused_data(self, viewer, fused, layer_name, cheap=False):
        # MVSRegistration.fuse() always returns msims, never sims - get_msim_level_data (each
        # level's raw dask array straight off its own Dataset) is always enough to show the
        # result in napari as a genuine multiscale pyramid, so nothing here needs a sim built
        # via extract_sims_from_fused. fused is either one real multiscale msim (a DataTree -
        # already channel-combined by fuse()'s own combine_msims_as_channels when there's more
        # than one channel, so a 'c' dim just needs channel_axis) or, in 'compose' mode (no
        # actual fusion), a plain list of per-source msims shown as separate layers.
        #
        # cheap=True (used by update_views()'s general overview, not preview_fusion()'s real
        # fusion-tab preview) skips get_contrast_limits()'s per-source dask.compute() in favour
        # of a naive dtype-range guess - correctness there doesn't matter for a first-look
        # overview, and it's the one per-source cost below that's neither Qt-only nor already
        # cheap metadata, so removing it (rather than threading it) is the actual win.
        channels = self.extra_metadata.get('channels', [])

        if isinstance(fused, list):
            # 'compose' mode: no real fusion, one separate napari layer per source. add_image()
            # itself must stay on the GUI thread (napari/Qt layers aren't thread-safe) - the part
            # a thread pool can still take off that thread is gathering each source's own
            # metadata/contrast-limits first, across every available core, joined synchronously
            # (results collected in original order) before the add_image() calls run serially.
            def prep_layer(msim, channel):
                image0 = get_msim_image0(msim)
                scale = si_utils.get_spacing_from_sim(image0, asarray=True)
                translate = si_utils.get_origin_from_sim(image0, asarray=True)
                contrast_limits = get_contrast_limits(msim, cheap=cheap)
                return dict(data=get_msim_level_data(msim), name=channel.get('label', layer_name),
                           multiscale=True, colormap=channel.get('color', (1, 1, 1, 1)),
                           contrast_limits=contrast_limits,
                           scale=scale, translate=translate, blending='additive')

            channel_list = channels or [{}] * len(fused)
            with Timer(f'_napari_view_add_fused_data: prep {len(fused)} layers',
                      verbose=self._timing_verbose()):
                layer_kwargs = [None] * len(fused)
                if len(fused) > 1:
                    max_workers = min(default_preview_workers, len(fused))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(prep_layer, msim, channel): index
                                  for index, (msim, channel) in enumerate(zip(fused, channel_list))}
                        for future in as_completed(futures):
                            layer_kwargs[futures[future]] = future.result()
                elif fused:
                    layer_kwargs[0] = prep_layer(fused[0], channel_list[0])

            with Timer(f'_napari_view_add_fused_data: add_image loop ({len(fused)} layers)',
                      verbose=self._timing_verbose()):
                for kwargs in layer_kwargs:
                    data = kwargs.pop('data')
                    viewer.add_image(data, **kwargs)
            return

        image0 = get_msim_image0(fused)
        scale = si_utils.get_spacing_from_sim(image0, asarray=True)
        translate = si_utils.get_origin_from_sim(image0, asarray=True)
        data = get_msim_level_data(fused)
        with Timer('_napari_view_add_fused_data: get_contrast_limits', verbose=self._timing_verbose()):
            contrast_limits = get_contrast_limits(fused, cheap=cheap)
        if len(channels) > 1 and 'c' in image0.dims:
            channel_axis = image0.dims.index('c')
            name = [channel.get('label', index) for index, channel in enumerate(channels)]
            colormap = [channel.get('color', (1, 1, 1, 1)) for channel in channels]
            scale = [scale] * len(channels)
            translate = [translate] * len(channels)
            contrast_limits = [contrast_limits] * len(channels)
        else:
            channel_axis = None
            name = channels[0].get('label') if channels else None
            colormap = channels[0].get('color', (1, 1, 1, 1)) if channels else None
        with Timer('_napari_view_add_fused_data: add_image', verbose=self._timing_verbose()):
            viewer.add_image(data, name=name or layer_name, multiscale=True, channel_axis=channel_axis,
                             colormap=colormap, contrast_limits=contrast_limits,
                             scale=scale, translate=translate)

    def _napari_view_show_features(self, viewer, fixed_data2, fixed_points, moving_data2, moving_points, matches, inliers):
        layers = draw_keypoints_matches_napari(fixed_data2, fixed_points,
                                               moving_data2, moving_points,
                                               matches, inliers, points_color='blue')
        viewer.layers.clear()
        for data, kwargs, layer_type in layers:
            if layer_type == "image":
                viewer.add_image(data, **kwargs)
            elif layer_type == "points":
                viewer.add_points(data, **kwargs)
            elif layer_type == "shapes":
                viewer.add_shapes(data, **kwargs)

    def _napari_view_add_image(self, viewer, data, label, transform=None, color=None, affine_event=False):
        if isinstance(data, DataTree):
            # a real multiscale msim (e.g. self.reg.register_msims) - napari's affine (used here
            # for interactive per-pair drag adjustment) and multiscale lazy-loading work together
            image0 = get_msim_image0(data)
            scale = si_utils.get_spacing_from_sim(image0, asarray=True)
            position = si_utils.get_origin_from_sim(image0, asarray=True)
            layer = viewer.add_image(get_msim_level_data(data), name=label, multiscale=True,
                                     scale=scale, translate=position, affine=transform,
                                     blending='additive')
        else:
            scale = si_utils.get_spacing_from_sim(data, asarray=True)
            position = si_utils.get_origin_from_sim(data, asarray=True)
            layer = viewer.add_image(data, name=label, scale=scale, translate=position, affine=transform,
                                     blending='additive')
        if color:
            layer.colormap = color

        if affine_event:
            layer.events.affine.connect(self.on_image_data_changed)

        return layer

    def on_image_data_changed(self, event):
        self.pair_metrics_timer.stop()
        self.pair_metrics_timer.start()

    def update_pair_metrics(self):
        # filter only selected pair
        reg_msims = [self.reg.register_msims[index] for index in self.pair_indices]
        transforms = {(0, 1): self.calc_mod_pair_transform()}
        metrics = calc_msims_metrics(reg_msims, transforms, metric_methods=self.metrics_methods)
        self.populate_metrics_table(metrics)

    @catch_run_errors
    def run_preview_registration(self):
        label1 = self.param_widgets.get('registration.reg_preview_image1').get_value()
        label2 = self.param_widgets.get('registration.reg_preview_image2').get_value()
        index1 = self.reg.file_labels.index(label1)
        index2 = self.reg.file_labels.index(label2)

        if not self.reg.register_msims:
            if not self.run_pre_processing():
                return None
        with NapariDaskProgress(progress_class=progress, desc='Preview registration'), \
                TemporarilyDisabledWidgets(self.enable_plugin_widget), \
                VisibleActivityDock(self.viewer):
            registration_params = self.params['registration']
            channel = registration_params.get('channel')
            cache = self._preview_overlap_cache
            # the overlap crop only depends on the source data (register_msims - a new list
            # object every time pre-processing actually re-runs) and which pair/channel is
            # selected, never on the registration method or its tuning parameters - reuse it
            # across parameter-only changes instead of re-cropping from the (possibly large)
            # source data every time
            if (cache is not None and cache['register_msims'] is self.reg.register_msims
                    and cache['index1'] == index1 and cache['index2'] == index2
                    and cache['channel'] == channel):
                overlap1, overlap2, sims_pixel_space = cache['overlap1'], cache['overlap2'], cache['sims_pixel_space']
            else:
                msim1, msim2 = self.reg.register_msims[index1], self.reg.register_msims[index2]
                overlap1, overlap2, sims_pixel_space = self.reg.select_pair_overlap(
                    msim1, msim2, params=registration_params)
                overlap1, overlap2 = overlap1.compute(), overlap2.compute()
                self._preview_overlap_cache = {
                    'register_msims': self.reg.register_msims,
                    'index1': index1, 'index2': index2, 'channel': channel,
                    'overlap1': overlap1, 'overlap2': overlap2, 'sims_pixel_space': sims_pixel_space,
                }

            transform, quality, results = self.reg.register_overlap(
                overlap1, overlap2, sims_pixel_space, params=registration_params)

            msim1, msim2 = self.reg.register_msims[index1], self.reg.register_msims[index2]
            transforms = {(0, 1): transform}
            qualities = {(0, 1): quality}
            metrics = calc_msims_metrics((msim1, msim2), transforms, qualities, metric_methods=self.metrics_methods)

        return metrics, results, overlap1, overlap2

    def preview_registration(self):
        self._clear_napari_view(self.viewer)
        result = self.run_preview_registration()
        if result is None:
            return
        metrics, results, overlap1, overlap2 = result

        self.populate_metrics_table(metrics)

        fixed_points = results.get('fixed_points', [])
        moving_points = results.get('moving_points', [])
        matches = results.get('matches', [])
        inliers = results.get('inliers', [])
        self._napari_view_show_features(self.viewer, overlap1, fixed_points, overlap2, moving_points, matches, inliers)
        self.view_mode = ViewMode.FEATURES

    def populate_metrics_table(self, metrics_dict):
        transform_keys = []
        metric_keys = []
        item_keys = []
        metrics = metrics_dict.get('summary')
        if metrics:
            item_keys.append('summary')
            for transform_key, transform_value in metrics.items():
                if transform_key not in transform_keys:
                    transform_keys.append(transform_key)
                for metric_key, metric_value in transform_value.items():
                    if metric_value is not None and metric_key not in metric_keys:
                        metric_keys.append(metric_key)
        metrics = metrics_dict.get('pairs')
        if metrics:
            for pair_key_indices, pair_value in metrics.items():
                pair_key = self.reg.file_labels[pair_key_indices[0]] + ' - ' + self.reg.file_labels[pair_key_indices[1]]
                if pair_key not in item_keys:
                    item_keys.append(pair_key)
                for transform_key, transform_value in pair_value.items():
                    if transform_key not in transform_keys:
                        transform_keys.append(transform_key)
                    for metric_key, metric_value in transform_value.items():
                        if metric_value is not None and metric_key not in metric_keys:
                            metric_keys.append(metric_key)

        transform_keys = [transform_key.split('_')[0] for transform_key in transform_keys]
        is_metric_cols = (len(transform_keys) <= 1 and len(metric_keys) >= 1)
        col_headers = metric_keys if is_metric_cols else transform_keys

        metrics_table = []
        for rowi in range(len(item_keys)):
            row = [None] * len(col_headers)
            metrics_table.append(row)
        item_offset = 0

        metrics = metrics_dict.get('summary')
        if metrics:
            item_offset = 1
            for transform_index, transform_value in enumerate(metrics.values()):
                for metric_index, metric_value in enumerate(transform_value.values()):
                    if metric_value is not None:
                        col_index = metric_index if is_metric_cols else transform_index
                        metrics_table[0][col_index] = metric_value
        metrics = metrics_dict.get('pairs')
        if metrics:
            for pair_index, pair_value in enumerate(metrics.values()):
                for transform_index, transform_value in enumerate(pair_value.values()):
                    for metric_index, metric_value in enumerate(transform_value.values()):
                        if metric_value is not None:
                            col_index = metric_index if is_metric_cols else transform_index
                            metrics_table[pair_index + item_offset][col_index] = metric_value

        table_widget = self.param_widgets.get('registration.metrics_table')
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((metrics_table, item_keys, col_headers))
        table_widget.set_table_column_resize_mode()
        for rowi in range(len(item_keys)):
            for coli in range(len(col_headers)):
                table_cell = table_widget.get_native_item(rowi, coli)
                if table_cell is not None:
                    table_cell.setBackground(
                        QColor(*metric_to_rgb(metrics_table[rowi][coli], max_light=0.5, output_range=255)))

    def update_registered(self, view_transform_key=None):
        msims = self.reg.msims
        coord_systems = get_transforms(msims)
        self.populate_coordinate_systems(coord_systems)
        self.populate_metadata_table(msims)
        self.populate_metrics_table(self.reg.metrics)
        self.update_views(transform_key=view_transform_key)

    def enable_modify_pair_registration(self, enabled=True):
        widget = self.param_widgets.get('registration.modify_pair_registration')
        if widget:
            widget.widget.enabled = enabled

    @catch_run_errors
    def run_pair_registration(self):
        if not self.reg.register_msims:
            if not self.run_pre_processing():
                return None

        with NapariMVSProgress(tqdm_class=progress, patch_registration=True), \
                NapariDaskProgress(progress_class=progress, desc='Pair registration'), \
                TemporarilyDisabledWidgets(self.enable_plugin_widget), \
                VisibleActivityDock(self.viewer), \
                Timer('pair registration', verbose=self._timing_verbose()):
            results = self.reg.register_pairs(self.reg.register_msims,
                                              params=self.params['registration'] | {'metrics': self.metrics_methods})

        qualities = {key: metric[default_transform_key][default_quality_key]
                     for key, metric in results['metrics']['pairs'].items()
                     if default_quality_key in metric[default_transform_key]}
        bboxes = {}
        for key, value in nx.get_edge_attributes(self.reg.pairs_graph, 'bbox').items():
            if 't' in value.dims:
                value = value.sel(t=0)
            bboxes[key] = np.array(value).tolist()
        self.reg.save_pair_mappings(results['pair_mappings'], qualities, bboxes)
        self.enable_modify_pair_registration()
        return results

    @catch_run_errors
    def run_global_registration(self):
        with NapariDaskProgress(progress_class=progress, desc='Global registration'), \
                TemporarilyDisabledWidgets(self.enable_plugin_widget), \
                VisibleActivityDock(self.viewer), \
                Timer('global registration', verbose=self._timing_verbose()):
            results = self.reg.register_global(self.reg.pair_msims,
                                               register_indices=self.reg.register_indices,
                                               params=self.params['registration'])

        self.reg.save_mappings(results['mappings'])
        self.reg.save_mappings_csv(results['mappings'])
        self.reg.save_metrics(results['metrics'])
        return results

    def pair_registration(self):
        if self.reg.is_global_registered():
            show_warning('Global registration was already performed')
        else:
            message = 'Pair registration was already performed. ' if self.reg.is_pairs_registered() else ''
            message += 'Run pair registration?'
            reply = QMessageBox.question(None, 'muvis-align', message,
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                if not self.run_pair_registration():
                    return
                self.update_registered(view_transform_key=self.reg.source_transform_key)
                QMessageBox.information(None, 'muvis-align', 'Pair registration completed')

    def modify_pair_registration(self):
        if self.view_mode == ViewMode.PAIRS:
            reply = QMessageBox.question(None, 'muvis-align','Store modified registration?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                # update transforms back into graph
                transform = self.calc_mod_pair_transform()
                pair_transforms = nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
                qualities = nx.get_edge_attributes(self.reg.pairs_graph, default_quality_key)
                if 't' in pair_transforms[self.pair_indices].dims:
                    transform = transform.expand_dims({'t': [0]})
                pair_transforms[self.pair_indices] = transform
                qualities[self.pair_indices] = np.array(1)    # set quality to 1
                nx.set_edge_attributes(self.reg.pairs_graph, pair_transforms, default_transform_key)
                nx.set_edge_attributes(self.reg.pairs_graph, qualities, default_quality_key)
                bboxes = {}
                for key, value in nx.get_edge_attributes(self.reg.pairs_graph, 'bbox').items():
                    if 't' in value.dims:
                        value = value.sel(t=0)
                    bboxes[key] = np.array(value).tolist()
                self.reg.save_pair_mappings(pair_transforms, qualities, bboxes)

            self.view_mode = ViewMode.OVERVIEW
            self.update_registered(view_transform_key=self.reg.source_transform_key)
            self.temp_widget_state.restore()
            if self.enable_tab:
                for section_id, was_enabled in self.temp_tab_states.items():
                    self.enable_tab(section_id, was_enabled)
        else:
            self.view_mode = ViewMode.PAIRS
            labels = self.reg.file_labels
            label1 = self.param_widgets.get('registration.reg_preview_image1').get_value()
            label2 = self.param_widgets.get('registration.reg_preview_image2').get_value()
            index1 = labels.index(label1)
            index2 = labels.index(label2)
            indices = index1, index2
            colors = [(0, 1, 0), (1, 0, 1)]     # green, purple
            pair_transforms = nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
            if indices not in pair_transforms and tuple(reversed(indices)) in pair_transforms:
                indices = tuple(reversed(indices))

            if indices not in pair_transforms:
                show_warning('No pair registration found for selected images')
            else:
                self.temp_widget_state = TemporarilyDisabledWidgets()
                all_widgets = self.get_all_widgets()
                all_widgets.pop('registration.modify_pair_registration', None)
                self.temp_widget_state.disable(all_widgets)
                if self.enable_tab and self.is_tab_enabled:
                    other_section_ids = [section_id for section_id in ['project'] + list(self.template.keys())
                                        if section_id != 'registration']
                    self.temp_tab_states = {section_id: self.is_tab_enabled(section_id)
                                            for section_id in other_section_ids}
                    for section_id in other_section_ids:
                        self.enable_tab(section_id, False)
                self.pair_indices = indices
                pair_transform = np.array(pair_transforms[indices].sel(t=0))
                eye = np.eye(max(pair_transform.shape))
                pair_transforms = pair_transform, eye

                if not self.reg.register_msims:
                    if not self.run_pre_processing():
                        return
                self._clear_napari_view(self.viewer)
                # register_msims is a real multiscale pyramid (built by preprocess()) - lets
                # napari lazily load whichever level it needs during interactive adjustment
                register_images = self.reg.register_msims
                for index, (sim_index, color) in enumerate(zip(indices, colors)):
                    self._napari_view_add_image(self.viewer, register_images[sim_index], labels[sim_index],
                                                pair_transforms[index], color, affine_event=True)
                self.update_pair_metrics()

    def calc_mod_pair_transform(self):
        transforms = [layer.affine.affine_matrix for layer in self.viewer.layers]
        matsize = len(si_utils.get_spatial_dims_from_sim(get_msim_image0(self.reg.msims[0]))) + 1
        transform = calculate_rigid_difference(transforms[1][-matsize:, -matsize:],
                                               transforms[0][-matsize:, -matsize:])
        return param_utils.affine_to_xaffine(transform)

    def registration_process(self):
        if 'convert' in self.params['registration']['operation']:
            # convert: each source written out individually at its own source/metadata
            # position, no registration and no fusion/blending - never reaches the fusion tab
            reply = QMessageBox.question(None, 'muvis-align', 'Convert data to OME-Zarr only?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.run_convert():
                    QMessageBox.information(None, 'muvis-align', 'Conversion completed')
            return

        completion_message = 'Global registration completed'
        if self.reg.is_global_registered():
            message = 'Global registration was already performed. Run global registration?'
        elif not self.reg.is_pairs_registered():
            message = 'Pair registration not performed yet. Run both pair and global registration?'
            completion_message = 'Registration completed'
        else:
            message = 'Run global registration?'
        reply = QMessageBox.question(None, 'muvis-align', message,
                                     QMessageBox.Yes|QMessageBox.No)
        if reply == QMessageBox.Yes:
            with Timer('registration process', verbose=self._timing_verbose()):
                if not self.reg.is_pairs_registered():
                    if not self.run_pair_registration():
                        return
                if not self.run_global_registration():
                    return
            copy_transforms_to_msims(self.reg.msims, self.view_msims, self.reg.reg_transform_key)
            self.update_registered(view_transform_key=self.reg.reg_transform_key)
            self.enable_tabs(True, 4)
            QMessageBox.information(None, 'muvis-align', completion_message)

    @catch_run_errors
    def preview_fusion(self):
        transform_key = self.reg.reg_transform_key
        with NapariMVSProgress(tqdm_class=progress, desc='Fusion', patch_fusion=True), \
             TemporarilyDisabledWidgets(self.enable_plugin_widget), \
             VisibleActivityDock(self.viewer):
            data = self._create_napari_data(transform_key, fusion_method=self.params['fusion']['method'])
        self._clear_napari_view(self.viewer)
        self._napari_view_add_fused_data(self.viewer, data, f'{self.reg.fileset_label} data')
        self.view_mode = ViewMode.FUSED
        # preview_fusion() is also what a resumed already-fused project draws (init_progress's
        # is_fused() branch) - unlike every other init_progress branch, it never otherwise goes
        # through update_views(), so the overview would stay empty without this
        self._refresh_overview_shapes(transform_key)

    @catch_run_errors
    def run_convert(self):
        operation = self.params['registration']['operation']
        output_folder = operation_to_past_participle(operation)
        ome_version = self.params['fusion']['ome_version']
        with NapariPreprocessProgress(progress_class=progress, desc='Initialising sources',
                                      bar_format=" ", min_duration=0.1) as progress_factory, \
             TemporarilyDisabledWidgets(self.enable_plugin_widget), \
             VisibleActivityDock(self.viewer):
            # see run_pre_processing() - builds msims with its own progress reporting instead of
            # silently as a side effect of the save loop below
            msims = self.reg.ensure_msims(progress_factory=progress_factory)

        with NapariPreprocessProgress(progress_class=progress, desc='Converting',
                                      min_duration=0.1) as progress_factory, \
             TemporarilyDisabledWidgets(self.enable_plugin_widget), \
             VisibleActivityDock(self.viewer), \
             Timer('convert', verbose=self._timing_verbose()):
            # MVSRegistration.fuse() isn't used here - even its 'compose' fusion_method still
            # builds one shared output_stack_properties canvas across every source first (wasted
            # work no per-source file needs), and is_channel_overlay would still combine sources
            # into a single multichannel image regardless of fusion_method whenever multiple
            # channels are configured. Each source is instead written out individually - one
            # file per input, named after it, into its own output subfolder, keeping its own
            # native pyramid levels exactly as-is (save_native_levels()) - never fused/resampled.
            #
            # Sequential, not a thread pool: to_ngff_zarr()/zarr's own async store internals
            # aren't safe to invoke concurrently from multiple threads each running their own
            # event loop (observed as a Windows PermissionError racing on a store's zarr.json
            # rename) - each write already parallelises its own array computation across every
            # core via dask's default threaded scheduler, which is enough on its own.
            #
            # A per-source count (files converted so far) is meaningful here, unlike
            # NapariDaskProgress's per-task count, which resets every iteration and says
            # nothing about how many of the sources are actually done.
            with progress_factory(total=len(msims), desc='Converting') as pbar:
                # file_labels (get_unique_file_labels()) are already disambiguated against every
                # other source - get_filetitle(filename) alone is not: two sources whose raw
                # filenames only differ by parent directory (e.g. S000/000_000.tiff vs
                # S001/000_000.tiff) would both title as "000_000" and silently overwrite each
                # other's output
                for label, position, msim in zip(self.reg.file_labels, self.reg.positions, msims):
                    output_filename = f'{output_folder}/{label}'
                    self.reg.save_native_levels(output_filename, msim, position=position,
                                                ome_version=ome_version)
                    pbar.update(1)
        return True

    @catch_run_errors
    def run_fusion(self):
        operation = self.params['registration']['operation']
        output_filename = operation_to_past_participle(operation)
        tile_size = self.params['fusion']['tile_size']
        if ',' in tile_size:
            tile_size = [int(size.strip()) for size in tile_size.split(',')]
        elif isinstance(tile_size, str):
            tile_size = int(tile_size.strip())
        with NapariMVSProgress(tqdm_class=progress, desc='Fusion', patch_fusion=True), \
             TemporarilyDisabledWidgets(self.enable_plugin_widget), \
             VisibleActivityDock(self.viewer), \
             Timer('fusion', verbose=self._timing_verbose()):
            fused_image, is_saved = self.reg.fuse(self.reg.msims,
                                                  fusion_method=self.params['fusion']['method'],
                                                  output_spacing=self.params['fusion']['spacing'],
                                                  dimension=self.params['input_output']['registration_dimension'],
                                                  output_filename=output_filename,
                                                  tile_size=tile_size,
                                                  ome_version=self.params['fusion']['ome_version'],
                                                  extra_metadata=self.extra_metadata)
            if not is_saved:
                # save() only accepts a single-resolution sim - fused_image is always the
                # whole multiscale pyramid now, so save its finest scale
                save_sim = extract_sims_from_fused(fused_image)
                self.reg.save(output_filename, save_sim,
                              transform_key=self.reg.reg_transform_key,
                              translations0=self.reg.positions,
                              channels=self.extra_metadata.get('channels', []),
                              tile_size=tile_size,
                              ome_version=self.params['fusion']['ome_version'])
        return fused_image

    def fusion_process(self):
        message = 'Fusion was already performed. ' if self.reg.is_fused() else ''
        message += 'Export fused data?'
        reply = QMessageBox.question(None, 'muvis-align', message,
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            fused_image = self.run_fusion()
            if fused_image is None:
                return
            self._clear_napari_view(self.viewer)
            self._napari_view_add_fused_data(self.viewer, fused_image, 'Fused')
            self.reg.state = RegState.FUSED
            self.view_mode = ViewMode.FUSED
            QMessageBox.information(None, 'muvis-align', 'Fusion completed')
