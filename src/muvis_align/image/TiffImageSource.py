import tifffile
from ngff_zarr import tiff_file_to_ngff_images, NgffMultiscales

from muvis_align.image.ome_tiff_helper import read_tiff_source_metadata
from muvis_align.util import convert_to_um
from muvis_align.image.ImageSource import ImageSource
from muvis_align.image.color_conversion import hexrgb_to_rgba


class TiffImageSource(ImageSource):
    def init_metadata(self):
        # Metadata only - the per-level arrays are built on first access to self.data
        # (_load_data below). Everything project load reads (per-level shapes and pixel sizes,
        # dtype, dimension order, origin, channels) is in the file's own tags and, for an
        # OME-TIFF, its XML; read_tiff_source_metadata() takes it straight off tifffile at
        # ~0.7ms per source, against ~8ms to obtain the same values from
        # tiff_file_to_ngff_images() - which gets there by opening tif.aszarr() and wrapping
        # every pyramid level in a dask array, work nothing has asked for yet.
        metadata = read_tiff_source_metadata(self.filename)
        if metadata is None:
            # ngff_zarr's private axis mapping is unavailable - fall back to letting it do the
            # whole job, arrays and all (_load_data then has nothing left to do)
            self._init_metadata_from_ngff_zarr()
            return

        self.dimension_order = metadata['dimension_order']
        self.shapes = metadata['shapes']
        self.shape = self.shapes[0]
        self.dtype = metadata['dtype']
        self.pixel_sizes = metadata['pixel_sizes']
        self.pixel_size = self.pixel_sizes[0]
        self.position = metadata['position']
        self.channels = metadata['channels']
        # TODO: check with RGB image if better approach is possible
        self.is_rgb = (self.get_nchannels() in (3, 4))
        self.rotation = 0

    def _load_data(self):
        self._data = self._read_ngff_images()[1]

    def _read_ngff_images(self):
        """(ngff_images, per-level arrays) from ngff_zarr - the reference reader, and still what
        produces the actual arrays. Only the first series is the image itself: later series (if
        any) are auxiliary (e.g. an embedded thumbnail/label image), not further pyramid levels.
        """
        ngff_image_data1 = tiff_file_to_ngff_images(self.filename, reuse_existing_pyramids=True)[0][1]
        if isinstance(ngff_image_data1, NgffMultiscales):
            ngff_images = ngff_image_data1.images
        else:
            ngff_images = [ngff_image_data1]
        return ngff_images, [ngff_image.data for ngff_image in ngff_images]

    def _init_metadata_from_ngff_zarr(self):
        # the original path, kept as the fallback for whenever the fast read declines
        ngff_images, datas = self._read_ngff_images()
        self.dimension_order = ''.join(ngff_images[0].dims)
        for index, ngff_image in enumerate(ngff_images):
            axes_units = ngff_image.axes_units or {}
            if index == 0 and ngff_image.channel_names:
                for channel_index, channel_name in enumerate(ngff_image.channel_names):
                    channel = {'label': channel_name}
                    if ngff_image.channel_colors:
                        channel['color'] = hexrgb_to_rgba(ngff_image.channel_colors[channel_index])
                    self.channels.append(channel)
            self.pixel_sizes.append({dim: convert_to_um(value, axes_units.get(dim, 'um'))
                                     for dim, value in ngff_image.scale.items() if dim in 'xyz'})
        self.data = datas
        self.dtype = datas[0].dtype
        self.shapes = [data.shape for data in datas]
        self.shape = self.shapes[0]
        self.is_rgb = (self.get_nchannels() in (3, 4))
        self.pixel_size = self.pixel_sizes[0]
        self.rotation = 0


def tags_to_dict(tags: tifffile.TiffTags) -> dict:
    tag_dict = {}
    for tag in tags.values():
        tag_dict[tag.name] = tag.value
    return tag_dict
