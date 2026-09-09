import logging

import dask.array as da
from multiview_stitcher import ngff_utils

from muvis_align.image.ImageSource import ImageSource
from muvis_align.image.color_conversion import hexrgb_to_rgba
from muvis_align.image.ome_zarr_util import read_ome_zarr_source_metadata


class ZarrImageSource(ImageSource):
    def init_metadata(self):
        # Metadata only - the per-level arrays are built on first access to self.data
        # (_load_data below), the same way TiffImageSource defers its own. Reading them eagerly
        # here meant every source paid for a full xarray DataTree of one sim per pyramid level,
        # plus one zarr.json read per level, during ordinary project load: ~100ms per source,
        # i.e. minutes across a few thousand sources, all of it for pixel-shaped data that
        # nothing has asked for yet (shapes, pixel sizes, origin and channels are all that
        # project load actually reads). See read_ome_zarr_source_metadata() for how those come
        # off the store's own consolidated metadata in a single read instead.
        metadata = read_ome_zarr_source_metadata(self.filename)

        self.dimension_order = metadata['dimension_order']
        self.shapes = metadata['shapes']
        self.shape = self.shapes[0]
        self.dtype = metadata['dtype']

        self.pixel_sizes = metadata['pixel_sizes']
        self.pixel_size = self.pixel_sizes[0]
        self.position = metadata['position']

        # the levels' own array paths within the store, for _load_data() - None whenever the
        # metadata read could not establish them, in which case _load_data() falls back to the
        # msim reader
        self._level_paths = metadata.get('paths')
        # how many levels the store itself has, before any are synthesized on top
        self._native_nlevels = len(self.shapes)

        # a 'c' dim is always present (si_utils.get_sim_from_array forces one, at size 1 for a
        # file that has none), so there is always at least one channel to describe. Labels stay
        # the plain channel indices the sim's own 'c' coords carry.
        self.channels = [{'label': str(index)} for index in range(metadata['nchannels'])]
        omero = metadata['omero']
        if omero:
            # previously read off self.msim.attrs, which read_msim_from_ome_zarr leaves empty -
            # so a store's omero channel colours never actually reached self.channels. They now
            # come from the store's own OME attributes, where they are really written.
            for channel, channel_metadata in zip(self.channels, omero.get('channels', [])):
                if channel_metadata.get('color'):
                    channel['color'] = hexrgb_to_rgba(channel_metadata['color'])

        self.rotation = 0

    def _load_data(self):
        """One dask array per level, opened straight off the store.

        The alternative - handing the whole job to ngff_utils.read_msim_from_ome_zarr and using
        its msim as-is (_build_msim_natively below) - costs the same to open but produces the
        msim itself, which means this source can offer no raw arrays, and a source with no raw
        arrays cannot have coarse levels synthesized for it (see _add_missing_pyramid_level_data).
        For a *single-resolution* store that is expensive: nothing downstream can then reduce
        resolution, so preview and pre-processing fuse at the store's native resolution however
        coarse a result was asked for. Measured on one real project: an 8x preview of
        single-resolution sources fused 64x the pixels per plane it needed to, turning a ~9 minute
        fusion into ~54 minutes. Going through self.data instead puts this on exactly the same
        footing as TiffImageSource - the base class builds the msim, and synthesizes the coarse
        levels a single-resolution store lacks.
        """
        if not self._level_paths:
            self._build_msim_natively()
            return
        try:
            datas = []
            for path, sim_shape in zip(self._level_paths, self.shapes):
                data = da.from_zarr(self.filename, component=path)
                if tuple(data.shape) != tuple(sim_shape):
                    # the store's own dims, mapped onto the sim's forced t/c order - that only
                    # ever inserts size-1 axes (ngff_dims_to_sim_dims keeps the file's own dim
                    # order), so a reshape is exactly that insertion and moves no data
                    data = data.reshape(sim_shape)
                datas.append(data)
        except Exception as e:
            # anything da.from_zarr cannot open by path on its own - a remote URL needing a
            # store/mapper, an unusual layout - stays correct by going back through the reader
            # that handled it before, just without synthesized coarse levels
            logging.warning(f'{self.filename}: could not open pyramid levels directly'
                            f' ({e}) - falling back to reading the msim')
            self._level_paths = None
            # the msim below has only the levels the store really has, so drop any this source
            # had already settled as synthesized (during __init__, back when opening the arrays
            # directly still looked possible) - otherwise self.shapes describes a deeper pyramid
            # than the msim has, and get_shape(level)/get_pixel_size(level) index past its end
            self.shapes = self.shapes[:self._native_nlevels]
            self.pixel_sizes = self.pixel_sizes[:self._native_nlevels]
            self.scale_factors = self.scale_factors[:self._native_nlevels]
            self._build_msim_natively()
            return
        self._data = datas

    def _build_msim_natively(self):
        # fallback for a store whose per-level array paths could not be established (see
        # init_metadata): read the msim via the same trusted reader as before, then re-stamp
        # only its transform. self._data stays empty, so no coarse levels are synthesized -
        # exactly the previous behaviour.
        self._msim = ngff_utils.read_msim_from_ome_zarr(self.filename, array_backend='dask',
                                                        transform_key=self.transform_key)
        self._restamp_msim()

    def _build_msim(self):
        # _load_data() may already have produced the msim itself (the fallback above), in which
        # case there is nothing left to build from self.data
        data = self.data
        if self._msim is not None:
            return
        if not data:
            raise ValueError(f'No image data available for {self.filename}')
        super()._build_msim()

    def _synthesized_level_factors(self):
        if not self._level_paths:
            # the msim comes from the native reader with only the levels the store really has,
            # and there are no raw arrays to synthesize others from - claiming extra levels here
            # would leave self.shapes describing a pyramid the msim does not have
            return []
        factors = super()._synthesized_level_factors()
        if factors:
            _warn_single_resolution_store(self.filename, len(factors))
        return factors


_warned_single_resolution = False


def _warn_single_resolution_store(filename, nlevels):
    """Once per run, whatever the source count - a project whose sources lack pyramids has
    thousands of them, and one line each would bury the log.

    Synthesizing the levels (strided subsampling of the finest one) is what keeps a coarse
    preview from fusing at native resolution, but unlike an already-decoded TIFF page it is not
    free here: each coarse level still reads every full-resolution chunk it strides across. It
    is the far cheaper of the two options - reading the same chunks once beats fusing 64x the
    output pixels - but re-converting the sources with a real pyramid avoids the re-read
    entirely.
    """
    global _warned_single_resolution
    if _warned_single_resolution:
        return
    _warned_single_resolution = True
    logging.warning(f'Single-resolution OME-Zarr source(s), e.g. {filename}: synthesizing'
                    f' {nlevels} coarse level(s) so a reduced-resolution preview/pre-processing'
                    f' does not have to fuse at the store\'s native resolution. Building them'
                    f' re-reads full-resolution chunks - re-converting these sources with a full'
                    f' pyramid avoids that.')
