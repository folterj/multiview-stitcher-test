from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.ImageSource import ImageSource
from muvis_align.image.color_conversion import hexrgb_to_rgba


class ZarrImageSource(ImageSource):
    def init_metadata(self):
        # keep the natively-read msim itself (self.msim) - ImageSource._restamp_msim() replaces
        # just its transform in place, instead of tearing it down to raw arrays for _build_msim()
        # to reconstruct a whole new msim from scratch. All metadata below is read directly off
        # each scale's own 'image' DataArray (self.msim[scale_key].ds['image']) - si_utils.
        # get_spacing_from_sim/get_origin_from_sim only ever look at .dims/.coords, so there's no
        # need to go through msi_utils.get_sim_from_msim (which additionally attaches transform
        # attrs we don't need here) just to read metadata.
        self._msim = ngff_utils.read_msim_from_ome_zarr(self.filename, array_backend='dask',
                                                        transform_key=self.transform_key)
        scale_keys = msi_utils.get_sorted_scale_keys(self.msim)
        images = [self.msim[scale_key].ds['image'] for scale_key in scale_keys]
        image0 = images[0]

        self.dimension_order = ''.join(image0.dims)
        self.shapes = [image.shape for image in images]
        self.shape = self.shapes[0]
        self.dtype = image0.dtype

        self.pixel_sizes = [si_utils.get_spacing_from_sim(image) for image in images]
        self.pixel_size = self.pixel_sizes[0]
        self.position = si_utils.get_origin_from_sim(image0)

        if 'c' in image0.dims:
            self.channels = [{'label': str(label)} for label in image0.coords['c'].values]
            omero = self.msim.attrs.get('omero')
            if omero:
                for channel, ch_meta in zip(self.channels, omero.get('channels', [])):
                    if 'color' in ch_meta:
                        channel['color'] = hexrgb_to_rgba(ch_meta['color'])

        self.rotation = 0
