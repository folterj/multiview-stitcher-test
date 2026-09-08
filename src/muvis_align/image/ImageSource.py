import logging
import os
from math import ceil

import numpy as np
from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.constants import default_transform_key
from muvis_align.image.util import (combine_transforms, build_source_redimensioned_msim,
                                    build_missing_pyramid_levels, calc_pyramid_level_factors)
from muvis_align.util import (find_all_numbers, split_numeric_dict, eval_context, check_contains_value,
                              create_transform, load_sbemimage_best_config, adjust_sbemimage_properties)


class ImageSource:
    default_physical_unit = 'µm'

    def __init__(self, filename, source_metadata=None, extra_metadata=None, file_label=None,
                 transform_key=default_transform_key, matrix_size=None):
        self.filename = filename
        self.file_label = file_label
        self.transform_key = transform_key
        self.dimension_order = ''
        self.is_rgb = False
        self.shapes = []
        self.shape = []
        self.dtype = None
        self.pixel_sizes = []
        self.pixel_size = {}
        self.scale_factors = []
        self.position = {}
        self.rotation = 0
        self.channels = []
        self._data = []
        self.metadata = {}
        self.transform = None
        self._msim = None
        self._redimensioned_msims = {}
        self._data_loaded = False
        self.init_metadata()
        self.fix_metadata(source_metadata, extra_metadata, matrix_size)
        # only the *metadata* for any synthesized coarse levels is settled here (pure shape
        # arithmetic, no arrays) - the arrays themselves are built with the rest of self.data,
        # on first access. Deciding it up front is what keeps get_shape/get_pixel_size/
        # get_level_from_scale answering the same thing however early they are called, rather
        # than growing extra levels the moment something first touches the pixel data.
        self._add_missing_pyramid_level_metadata()

    @property
    def msim(self):
        if self._msim is None:
            self._build_msim()
        return self._msim

    @property
    def data(self):
        """One array per pyramid level - loaded on first access, never during __init__.

        Project load only ever reads metadata (shapes, pixel sizes, position, channels), so a
        source that opened its arrays eagerly made every run pay for pixel-shaped data nothing
        had asked for yet: measured at ~100ms per source for OME-Zarr and ~8ms for TIFF, i.e.
        minutes across a few thousand sources.
        """
        if not self._data_loaded:
            self._data_loaded = True
            self._load_data()
            self._add_missing_pyramid_level_data()
        return self._data

    @data.setter
    def data(self, value):
        # a subclass that populates self.data straight from init_metadata()/_load_data() sets it
        # through here; going through the property keeps _load_data() from running afterwards
        # and overwriting it
        self._data_loaded = True
        self._data = value

    def get_msim(self, output_order):
        """self.msim redimensioned to `output_order`, built once and cached per output_order -
        build_source_msim() calls this on every run instead of redimensioning self.msim from
        scratch each time, since redimensioning only depends on (self, output_order), never on
        per-run geometry (translation/transform).
        """
        if output_order not in self._redimensioned_msims:
            self._redimensioned_msims[output_order] = build_source_redimensioned_msim(self, output_order)
        return self._redimensioned_msims[output_order]

    def init_metadata(self):
        raise NotImplementedError("Image source should implement init_metadata() to initialize metadata,"
                                  " populating self.data with one dask array per pyramid level")

    def fix_metadata(self, source_metadata=None, extra_metadata=None, matrix_size=None):
        if isinstance(source_metadata, dict):
            filename_numeric = find_all_numbers(self.filename)
            filename_dict = {key: int(value) for key, value in split_numeric_dict(self.filename).items()}
            context = {'filename_numeric': filename_numeric, 'fn': filename_numeric} | filename_dict
            if 'position' in source_metadata:
                translation = source_metadata['position']
                if 'x' in translation:
                    if not check_contains_value(translation['x'], 'source'):
                        self.position['x'] = eval_context(translation, 'x', 0, context)
                    if check_contains_value(translation['x'], 'invert'):
                        if isinstance(self.position['x'], (tuple, list)):
                            self.position['x'] = -self.position['x'][0], self.position['x'][1]
                        else:
                            self.position['x'] = -self.position['x']
                if 'y' in translation:
                    if not check_contains_value(translation['y'], 'source'):
                        self.position['y'] = eval_context(translation, 'y', 0, context)
                    if check_contains_value(translation['y'], 'invert'):
                        if isinstance(self.position['y'], (tuple, list)):
                            self.position['y'] = -self.position['y'][0], self.position['y'][1]
                        else:
                            self.position['y'] = -self.position['y']
                if 'z' in translation:
                    if not check_contains_value(translation['z'], 'source'):
                        self.position['z'] = eval_context(translation, 'z', 0, context)
                    if check_contains_value(translation['z'], 'invert'):
                        if isinstance(self.position['z'], (tuple, list)):
                            self.position['z'] = -self.position['z'][0], self.position['z'][1]
                        else:
                            self.position['z'] = -self.position['z']
            if 'scale' in source_metadata:
                scale = source_metadata['scale']
                if 'x' in scale:
                    if not check_contains_value(scale['x'], 'source'):
                        self.pixel_sizes[0]['x'] = eval_context(scale, 'x', 1, context)
                if 'y' in scale:
                    if not check_contains_value(scale['y'], 'source'):
                        self.pixel_sizes[0]['y'] = eval_context(scale, 'y', 1, context)
                if 'z' in scale:
                    if not check_contains_value(scale['z'], 'source'):
                        self.pixel_sizes[0]['z'] = eval_context(scale, 'z', 1, context)
            if 'rotation' in source_metadata:
                if not check_contains_value(source_metadata['rotation'], 'source'):
                    self.rotation = eval_context(source_metadata, 'rotation', 0, context)
                if check_contains_value(source_metadata['rotation'], 'invert'):
                    self.rotation = -self.rotation
            if 'channels' in source_metadata and source_metadata['channels']:
                self.channels = source_metadata['channels']

        self.scale_factors = [{dim: value0 / value for dim, value, value0
                               in zip(self.dimension_order, shape, self.shape) if dim in 'xyz'}
                               for shape in self.shapes]
        # re-derive every level's pixel size from the (possibly overridden) level 0 value and the
        # shape-based scale factors, so a 'scale' override in source_metadata actually takes effect
        # (previously it only ever touched a self.pixel_size dict that get_pixel_size() never read)
        self.pixel_sizes = [{dim: self.pixel_sizes[0][dim] * factor.get(dim, 1) for dim in self.pixel_sizes[0]}
                            for factor in self.scale_factors]
        self.pixel_size = self.pixel_sizes[0]

        if isinstance(source_metadata, dict):
            if 'is_center' in source_metadata:
                self.position = {dim: self.position[dim] - self.get_physical_size().get(dim, 0) / 2
                                 for dim in self.position}

            if 'sbem' in source_metadata:
                source_version = self.metadata.get('Creator', self.metadata.get('creator', ''))
                if '2025' in source_version:
                    path = os.path.dirname(self.filename)
                    metapath = None
                    attempts = 0
                    while attempts < 3:
                        metapath = os.path.join(path, 'meta')
                        if os.path.exists(metapath):
                            break
                        path = os.path.join(path, '..')
                        attempts += 1
                    if metapath:
                        sbemimage_config = load_sbemimage_best_config(metapath, self.filename)
                        if sbemimage_config:
                            size = self.get_size()
                            translation, scale0 = adjust_sbemimage_properties(
                                self.position, self.pixel_size, size, self.filename, sbemimage_config)
                            self.position = translation
                            if scale0:
                                self.pixel_sizes[0] = scale0
                                self.pixel_size = self.pixel_sizes[0]
                            elif self.pixel_size.get('x') != self.pixel_size.get('y'):
                                logging.warning('SBEMimage pixel size requires correction,'
                                                ' please provide in source metadata.')
                            logging.debug(f'Adjusted SBEMimage properties for {self.filename}')
                        else:
                            logging.warning(f'Could not find SBEMimage config for {self.filename}.')
                    else:
                        logging.warning(f'Could not find SBEMimage config for {self.filename}.')

            if 'invert' in source_metadata:
                if 'x' in self.position:
                    self.position['x'] = -self.position['x']
                if 'y' in self.position:
                    self.position['y'] = -self.position['y']

        if matrix_size is None:
            matrix_size = 4 if self.get_size().get('z', 0) > 1 else 3
        if self.rotation is None:
            self.transform = None
        else:
            self.transform = param_utils.invert_coordinate_order(
                create_transform(self.position, self.rotation, matrix_size=matrix_size))
        if extra_metadata and self.file_label is not None and self.file_label in extra_metadata:
            transform2 = np.array(extra_metadata[self.file_label])
            if self.transform is None:
                self.transform = transform2
            else:
                self.transform = np.array(combine_transforms([self.transform, transform2]))

    def _load_data(self):
        """Populate self._data with one array per level of the source's own pyramid. Called once,
        on first access to self.data - a subclass with no raw arrays to offer (e.g.
        ZarrImageSource, whose msim is read natively) simply leaves it empty.
        """

    def _synthesized_level_factors(self):
        """Cumulative downsample factors of the coarse levels this source has to synthesize -
        empty unless it is single-resolution (a plain, non-pyramidal TIFF).

        A source with only one real resolution leaves napari with no coarse level to show while
        zoomed out, so drawing it forces computing the *entire* finest-level dask graph just to
        render a thumbnail-sized view - the usual cause of a slow first draw despite loading
        (building the lazy graph) itself being fast. Only formats that can subsample that one
        level cheaply qualify: build_missing_pyramid_levels() slices an already-decoded page, so
        it is near-free there, whereas a chunked store would have to read every chunk it touches
        and gain nothing (hence ZarrImageSource offers no raw arrays at all - see _load_data).
        """
        if len(self.shapes) != 1:
            return []
        sizes = {dim: size for dim, size in zip(self.dimension_order, self.shape) if dim in 'xyz'}
        return calc_pyramid_level_factors(sizes)

    def _add_missing_pyramid_level_metadata(self):
        # shapes/pixel_sizes/scale_factors for the levels _add_missing_pyramid_level_data() will
        # later build, derived from the finest level's shape alone - no arrays needed, so this
        # settles during __init__ while the arrays themselves stay deferred. _build_msim() (and
        # get_shape/get_pixel_size etc.) index these by level.
        axes = {dim: axis for axis, dim in enumerate(self.dimension_order)}
        for level_factors in self._synthesized_level_factors():
            shape = tuple(ceil(size / level_factors.get(dim, 1))
                          for dim, size in zip(self.dimension_order, self.shape))
            self.shapes.append(shape)
            # from the realised extent ratio, not the requested factor: strided slicing rounds
            # up, so a small or odd extent shrinks by less than its factor (z of 3 halved twice
            # is 1, a ratio of 3, not 4) and the pixel size has to follow the pixels
            self.pixel_sizes.append({dim: value * self.shape[axes[dim]] / shape[axes[dim]]
                                     for dim, value in self.pixel_sizes[0].items()
                                     if dim in axes})
        if len(self.shapes) > 1:
            # keep scale_factors (as set by fix_metadata(), for get_level_from_scale()) in sync
            # with the levels just added, rather than leaving it at its single-level value
            self.scale_factors = [{dim: value0 / value for dim, value, value0
                                   in zip(self.dimension_order, shape, self.shape) if dim in 'xyz'}
                                   for shape in self.shapes]

    def _add_missing_pyramid_level_data(self):
        # the array half of the above, run once self._data actually exists
        if len(self._data) != 1 or len(self.shapes) == 1:
            return
        datas, _ = build_missing_pyramid_levels(
            self._data[0], self.dimension_order, self.pixel_sizes[0])
        # both halves come from the same rule (calc_pyramid_level_factors), so they must agree -
        # a mismatch would leave self.data and self.shapes describing different pyramids
        assert [tuple(data.shape) for data in datas] == [tuple(shape) for shape in self.shapes], \
            (f'{self.filename}: synthesized levels {[tuple(d.shape) for d in datas]} do not'
             f' match the shapes settled at init {self.shapes}')
        self._data = datas

    def _build_msim(self):
        # si_utils.get_sim_from_array forces a 'c' dim regardless (size 1 if not already in
        # dimension_order) - label it unconditionally so a channel selected by name (e.g.
        # registration's 'channel' param) can be found via .sel(c=...) either way
        c_coords = [channel.get('label', '') for channel in self.get_channels()]
        # fix empty/incomplete dicts: si_utils.get_sim_from_array requires either a translation
        # covering every spatial dim, or None (letting it default everything to 0)
        translation = dict(self.position)
        if translation:
            if 'x' not in translation:
                translation['x'] = 0
            if 'y' not in translation:
                translation['y'] = 0
        translation_arg = translation if translation else None
        sims = []
        for level, data in enumerate(self.data):
            scale = self.pixel_sizes[level]
            scale_arg = scale if scale else None
            sims.append(si_utils.get_sim_from_array(
                data, dims=list(self.dimension_order),
                scale=scale_arg, translation=translation_arg,
                affine=self.transform, transform_key=self.transform_key,
                c_coords=c_coords))
        self._msim = msi_utils.get_msim_from_sims(sims)

    def _restamp_msim(self):
        # a subclass (e.g. ZarrImageSource) already built self.msim natively via a trusted
        # reader (e.g. read_msim_from_ome_zarr) - its own position/pixel_size stay exactly as
        # read, since nothing in the pipeline ever reads geometry off the msim's own coords
        # (MVSRegistration always uses get_position()/get_pixel_size() instead). The one thing
        # that does need replacing is the transform: such readers set an identity transform in
        # their own convention (e.g. read_msim_from_ome_zarr always uses 4x4, since z counts as
        # a real spatial dim in NGFF even at size 1), which must become our own self.transform,
        # in our own convention (matching si_utils.get_sim_from_array: no 't' dim, sized to only
        # the spatial dims that matter here).
        if self.transform is not None:
            xaffine = param_utils.affine_to_xaffine(self.transform)
        else:
            spatial_dims = [dim for dim in self.dimension_order if dim in 'xyz']
            xaffine = param_utils.identity_transform(len(spatial_dims))
        for scale_key in msi_utils.get_sorted_scale_keys(self.msim):
            ds = self.msim[scale_key].to_dataset()
            # drop the old x_in/x_out coordinate index too, not just the transform variable -
            # otherwise a differently-shaped replacement gets reindexed/padded with NaN against it
            ds = ds.drop_vars([self.transform_key, 'x_in', 'x_out'], errors='ignore')
            self.msim[scale_key] = ds.assign({self.transform_key: xaffine})

    def get_level_data(self, level=0):
        # raw dask array straight off the msim's own DataTree node - no sim wrapping needed
        return self.msim[f'scale{level}'].ds['image'].data

    def get_shape(self, level=0):
        # shape in pixels
        return self.shapes[level]

    def get_size(self, level=0, asarray=False, axes='zyx'):
        # size in pixels
        size = {dim: size for dim, size in zip(self.dimension_order, self.get_shape(level))}
        if asarray:
            return np.array([size[dim] for dim in axes if dim in size])
        else:
            return size

    def get_pixel_size(self, level=0, asarray=False, axes='zyx'):
        # pixel size in micrometers
        pixel_size = self.pixel_sizes[level]
        if asarray:
            return np.array([pixel_size[dim] for dim in axes if dim in pixel_size])
        else:
            return pixel_size

    def get_physical_size(self, asarray=False, axes='zyx'):
        pixel_size = self.get_pixel_size()
        size = self.get_size()
        physical_size = {dim: size[dim] * pixel_size[dim] for dim in size if dim in pixel_size}
        if asarray:
            return np.array([physical_size[dim] for dim in axes if dim in physical_size])
        else:
            return physical_size

    def get_position(self, asarray=False, axes='zyx'):
        # position in micrometers
        if asarray:
            return np.array([self.position[dim] for dim in axes if dim in self.position])
        else:
            return self.position

    def get_rotation(self):
        # rotation in degrees
        return self.rotation

    def get_nchannels(self):
        return self.get_size().get('c', 1)

    def get_channels(self):
        if not self.channels:
            if self.is_rgb:
                return [{'label': ''}]
            else:
                return [{'label': f'channel {index}'} for index in range(self.get_nchannels())]
        return self.channels
