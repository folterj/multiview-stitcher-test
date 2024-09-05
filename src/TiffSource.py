# https://pypi.org/project/tifffile/

import dask.array as da
from enum import Enum
import numpy as np
import os
from tifffile import TiffFile, TiffPage, PHOTOMETRIC

from OmeSource import OmeSource
from image.util import *
from util import *


class TiffSource(OmeSource):
    """Tiff-compatible image source"""

    filename: str
    """original filename"""
    pages: list
    """list of all relevant TiffPages"""
    data: bytes
    """raw un-decoded image byte data"""
    arrays: list
    """list of all image arrays for different sizes"""

    def __init__(self,
                 filename: str,
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()
        self.data = bytes()
        self.arrays = []
        photometric = None
        nchannels = 1

        tiff = TiffFile(filename)
        self.tiff = tiff
        self.first_page = tiff.pages.first
        if tiff.is_ome and tiff.ome_metadata is not None:
            xml_metadata = tiff.ome_metadata
            self.metadata = tifffile.xml2dict(xml_metadata)
            if 'OME' in self.metadata:
                self.metadata = self.metadata['OME']
                self.has_ome_metadata = True
            if 'BinaryOnly' in self.metadata:
                # binary image only; get metadata from metadata file instead
                self.has_ome_metadata = False
                metdata_filename = os.path.join(os.path.dirname(filename), self.metadata['BinaryOnly'].get('MetadataFile'))
                if os.path.isfile(metdata_filename):
                    metdata_tiff = TiffFile(metdata_filename)
                    if metdata_tiff.is_ome and metdata_tiff.ome_metadata is not None:
                        xml_metadata = metdata_tiff.ome_metadata
                        self.metadata = tifffile.xml2dict(xml_metadata)
                        if 'OME' in self.metadata:
                            self.metadata = self.metadata['OME']
                            images = self.metadata.get('Image')
                            if isinstance(images, list):
                                for image in images:
                                    if image.get('Name', '').lower() == get_filetitle(filename).lower():
                                        self.metadata['Image'] = image
                                        break
                            self.has_ome_metadata = True

        if self.has_ome_metadata:
            pass
        elif tiff.is_imagej:
            self.metadata = tiff.imagej_metadata
        elif self.first_page.description:
            self.metadata = desc_to_dict(self.first_page.description)
        self.tags = tags_to_dict(self.first_page.tags)
        if 'FEI_TITAN' in self.tags:
            metadata = tifffile.xml2dict(self.tags.pop('FEI_TITAN'))
            if 'FeiImage' in metadata:
                metadata = metadata['FeiImage']
            self.metadata.update(metadata)

        if tiff.series:
            series0 = tiff.series[0]
            self.dimension_order = series0.axes
            photometric = series0.keyframe.photometric
        self.pages = get_tiff_pages(tiff)
        for page0 in self.pages:
            if isinstance(page0, list):
                page = page0[0]
                npages = len(page0)
            else:
                page = page0
                npages = 1
            self.npages = npages
            if not self.dimension_order:
                self.dimension_order = page.axes
                photometric = page.photometric
            shape = page.shape
            nchannels = shape[2] if len(shape) > 2 else 1
            nt = 1
            if isinstance(page, TiffPage):
                width = page.imagewidth
                height = page.imagelength
                self.depth = page.imagedepth
                depth = self.depth * npages
                bitspersample = page.bitspersample
            else:
                width = shape[1]
                height = shape[0]
                depth = npages
                if len(shape) > 2:
                    self.depth = shape[2]
                    depth *= self.depth
                bitspersample = page.dtype.itemsize * 8
            if self.has_ome_metadata:
                pixels = ensure_list(self.metadata.get('Image', {}))[0].get('Pixels', {})
                depth = int(pixels.get('SizeZ', depth))
                nchannels = int(pixels.get('SizeC', nchannels))
                nt = int(pixels.get('SizeT', nt))
            self.sizes.append((width, height))
            self.sizes_xyzct.append((width, height, depth, nchannels, nt))
            self.pixel_types.append(page.dtype)
            self.pixel_nbits.append(bitspersample)

        self.fh = tiff.filehandle
        self.dimension_order = self.dimension_order.lower().replace('s', 'c')

        self.is_rgb = (photometric in (PHOTOMETRIC.RGB, PHOTOMETRIC.PALETTE) and nchannels in (3, 4))

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

    def _find_metadata(self):
        pixel_size = []
        # from OME metadata
        if self.has_ome_metadata:
            self._get_ome_metadata()
            return

        # from imageJ metadata
        pixel_size_z = None
        pixel_size_unit = self.metadata.get('unit', '').encode().decode('unicode_escape')
        if pixel_size_unit == 'micron':
            pixel_size_unit = self.default_physical_unit
        if 'scales' in self.metadata:
            for scale in self.metadata['scales'].split(','):
                scale = scale.strip()
                if scale != '':
                    pixel_size.append((float(scale), pixel_size_unit))
        if len(pixel_size) == 0 and self.metadata is not None and 'spacing' in self.metadata:
            pixel_size_z = (self.metadata['spacing'], pixel_size_unit)
        # from description
        if len(pixel_size) < 2 and 'pixelWidth' in self.metadata:
            pixel_info = self.metadata['pixelWidth']
            pixel_size.append((pixel_info['value'], pixel_info['unit']))
            pixel_info = self.metadata['pixelHeight']
            pixel_size.append((pixel_info['value'], pixel_info['unit']))
        if len(pixel_size) < 2 and 'MPP' in self.metadata:
            pixel_size.append((self.metadata['MPP'], self.default_physical_unit))
            pixel_size.append((self.metadata['MPP'], self.default_physical_unit))
        # from page TAGS
        if len(pixel_size) < 2:
            pixel_size_unit = self.tags.get('ResolutionUnit', '')
            if isinstance(pixel_size_unit, Enum):
                pixel_size_unit = pixel_size_unit.name
            pixel_size_unit = pixel_size_unit.lower()
            if pixel_size_unit == 'none':
                pixel_size_unit = ''
            res0 = convert_rational_value(self.tags.get('XResolution'))
            if res0 is not None and res0 != 0:
                pixel_size.append((1 / res0, pixel_size_unit))
            res0 = convert_rational_value(self.tags.get('YResolution'))
            if res0 is not None and res0 != 0:
                pixel_size.append((1 / res0, pixel_size_unit))

        position = []
        xpos = convert_rational_value(self.tags.get('XPosition'))
        ypos = convert_rational_value(self.tags.get('YPosition'))
        if xpos is not None and ypos is not None:
            position = [(xpos, pixel_size_unit), (ypos, pixel_size_unit)]

        if pixel_size_z is not None and len(pixel_size) == 2:
            pixel_size.append(pixel_size_z)

        mag = self.metadata.get('Mag', self.metadata.get('AppMag', 0))

        nchannels = self.get_nchannels()
        photometric = str(self.metadata.get('PhotometricInterpretation', '')).lower().split('.')[-1]
        if nchannels == 3:
            channels = [{'label': photometric}]
        else:
            channels = [{'label': photometric}] * nchannels

        self.source_pixel_size = pixel_size
        self.source_mag = mag
        self.channels = channels
        self.position = position

    def get_source_dask(self):
        return self._load_as_dask()

    def _load_as_dask(self):
        if len(self.arrays) == 0:
            for level in range(len(self.sizes)):
                data = da.from_zarr(self.tiff.aszarr(level=level))
                if data.chunksize == data.shape:
                    data = data.rechunk()
                self.arrays.append(data)
        return self.arrays

    def _asarray_level(self, level: int, **slicing) -> np.ndarray:
        self._load_as_dask()
        redim = redimension_data(self.arrays[level], self.dimension_order, self.get_dimension_order())
        slices = get_numpy_slicing(self.get_dimension_order(), **slicing)
        out = redim[slices]
        return out
