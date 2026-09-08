from multiview_stitcher import ngff_utils

from muvis_align.image.ImageSource import ImageSource
from muvis_align.image.color_conversion import hexrgb_to_rgba
from muvis_align.image.ome_zarr_util import read_ome_zarr_source_metadata


class ZarrImageSource(ImageSource):
    def init_metadata(self):
        # Metadata only - self.msim is built on first access (_build_msim below), the same way
        # TiffImageSource defers its own. Reading it eagerly here meant every source paid for a
        # full xarray DataTree of one sim per pyramid level, plus one zarr.json read per level,
        # during ordinary project load: ~100ms per source, i.e. minutes across a few thousand
        # sources, all of it for pixel-shaped data that nothing has asked for yet (shapes,
        # pixel sizes, origin and channels are all that project load actually reads). See
        # read_ome_zarr_source_metadata() for how those come off the store's own consolidated
        # metadata in a single read instead.
        metadata = read_ome_zarr_source_metadata(self.filename)

        self.dimension_order = metadata['dimension_order']
        self.shapes = metadata['shapes']
        self.shape = self.shapes[0]
        self.dtype = metadata['dtype']

        self.pixel_sizes = metadata['pixel_sizes']
        self.pixel_size = self.pixel_sizes[0]
        self.position = metadata['position']

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

    def _build_msim(self):
        # Overrides ImageSource._build_msim (which reconstructs a msim from self.data, the raw
        # per-level arrays a reader like this one never populates): read the msim natively
        # instead, via the same trusted reader as before, then re-stamp only its transform -
        # exactly what __init__ used to do eagerly, just deferred to the first real use.
        self._msim = ngff_utils.read_msim_from_ome_zarr(self.filename, array_backend='dask',
                                                        transform_key=self.transform_key)
        self._restamp_msim()

    def _synthesized_level_factors(self):
        # Never synthesize coarse levels for a chunked store, whatever its pyramid depth: the
        # strided subsampling that makes this near-free for an already-decoded TIFF page still
        # reads every chunk it touches here, so a "coarse" level would cost the same as the full
        # one and buy nothing. A single-resolution OME-Zarr is a file that needs re-converting
        # with a full pyramid (select_msim_subpyramid_at_scale warns when one falls short).
        return []
