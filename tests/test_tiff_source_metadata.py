"""TiffImageSource must read metadata without building arrays - and get identical answers.

Project load only reads per-level shapes and pixel sizes, dtype, dimension order, origin and
channels. Obtaining those from ngff_zarr.tiff_file_to_ngff_images() costs ~8ms per source
because it opens tif.aszarr() and wraps every pyramid level in a dask array (~18 round-trips
through zarr's async/sync bridge), and for an OME-TIFF additionally DOM-parses the whole OME XML
- which a multi-file OME-TIFF repeats in every file's header.

The fast read must agree exactly with that reference, so the equivalence is asserted here across
every file shape a source can arrive in, rather than argued from the TIFF/OME specs.
"""
import numpy as np
import pytest
import tifffile

from muvis_align.image.ome_tiff_helper import (extract_ome_image_metadata,
                                               read_tiff_source_metadata)
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.TiffImageSource import TiffImageSource


def reference_metadata(path):
    """What init_metadata used to derive, via ngff_zarr."""
    from muvis_align.util import convert_to_um

    source = TiffImageSource.__new__(TiffImageSource)
    source.filename = path
    source.channels = []
    source.pixel_sizes = []
    source._data = []
    source._data_loaded = False
    source.shapes = []
    source.is_rgb = False
    source.position = {}
    source.dimension_order = ''
    source._init_metadata_from_ngff_zarr()
    return {'dimension_order': source.dimension_order,
            'shapes': [tuple(shape) for shape in source.shapes],
            'dtype': source.dtype,
            'pixel_sizes': source.pixel_sizes,
            'channels': source.channels,
            'is_rgb': source.is_rgb}


def compare(fast, reference):
    assert fast['dimension_order'] == reference['dimension_order']
    assert [tuple(shape) for shape in fast['shapes']] == reference['shapes']
    assert fast['dtype'] == reference['dtype']
    assert len(fast['pixel_sizes']) == len(reference['pixel_sizes'])
    for got, want in zip(fast['pixel_sizes'], reference['pixel_sizes']):
        assert set(got) == set(want)
        for dim in want:
            assert got[dim] == pytest.approx(want[dim])
    assert fast['channels'] == reference['channels']


def write_plain(path, shape, dtype=np.uint16, **kwargs):
    tifffile.imwrite(str(path), np.zeros(shape, dtype=dtype), **kwargs)
    return str(path)


def write_pyramid(path, shape, levels=3, dtype=np.uint16, **kwargs):
    data = np.zeros(shape, dtype=dtype)
    with tifffile.TiffWriter(str(path)) as writer:
        writer.write(data, subifds=levels - 1, tile=(256, 256), **kwargs)
        for level in range(1, levels):
            step = 2 ** level
            writer.write(data[..., ::step, ::step], subfiletype=1, tile=(256, 256))
    return str(path)


def test_plain_2d(tmp_path):
    path = write_plain(tmp_path / 'plain.tiff', (1024, 768))
    compare(read_tiff_source_metadata(path), reference_metadata(path))


def test_pyramidal_2d(tmp_path):
    path = write_pyramid(tmp_path / 'pyr.tiff', (2048, 2048))
    fast = read_tiff_source_metadata(path)
    assert len(fast['shapes']) == 3
    compare(fast, reference_metadata(path))


def test_uint8_and_float_dtypes(tmp_path):
    for index, dtype in enumerate((np.uint8, np.float32)):
        path = write_plain(tmp_path / f'dt{index}.tiff', (512, 512), dtype=dtype)
        compare(read_tiff_source_metadata(path), reference_metadata(path))


def test_rgb_lands_on_a_channel_dim(tmp_path):
    # tifffile reports 'YXS' (samples); ngff_zarr maps S onto 'c', so is_rgb must still hold
    path = write_plain(tmp_path / 'rgb.tiff', (512, 512, 3), dtype=np.uint8, photometric='rgb')
    reference = reference_metadata(path)
    fast = read_tiff_source_metadata(path)
    compare(fast, reference)
    assert 'c' in fast['dimension_order']

    source = create_image_source(path)
    assert source.is_rgb is reference['is_rgb'] is True


def test_multichannel_ome(tmp_path):
    path = str(tmp_path / 'multi.ome.tiff')
    data = np.zeros((3, 256, 256), dtype=np.uint16)
    tifffile.imwrite(path, data, metadata={'axes': 'CYX',
                                           'PhysicalSizeX': 0.25, 'PhysicalSizeXUnit': 'µm',
                                           'PhysicalSizeY': 0.25, 'PhysicalSizeYUnit': 'µm',
                                           'Channel': {'Name': ['DAPI', 'GFP', 'RFP']}})
    compare(read_tiff_source_metadata(path), reference_metadata(path))
    fast = read_tiff_source_metadata(path)
    assert [channel['label'] for channel in fast['channels']] == ['DAPI', 'GFP', 'RFP']
    assert fast['pixel_sizes'][0] == pytest.approx({'y': 0.25, 'x': 0.25})


def test_3d_ome_with_z_spacing(tmp_path):
    path = str(tmp_path / 'z.ome.tiff')
    data = np.zeros((4, 256, 256), dtype=np.uint16)
    tifffile.imwrite(path, data, metadata={'axes': 'ZYX',
                                           'PhysicalSizeX': 0.5, 'PhysicalSizeXUnit': 'µm',
                                           'PhysicalSizeY': 0.5, 'PhysicalSizeYUnit': 'µm',
                                           'PhysicalSizeZ': 2.0, 'PhysicalSizeZUnit': 'µm'})
    fast = read_tiff_source_metadata(path)
    compare(fast, reference_metadata(path))
    assert fast['pixel_sizes'][0] == pytest.approx({'z': 2.0, 'y': 0.5, 'x': 0.5})


def test_millimetre_units_are_converted(tmp_path):
    path = str(tmp_path / 'mm.ome.tiff')
    tifffile.imwrite(path, np.zeros((256, 256), dtype=np.uint16),
                     metadata={'axes': 'YX',
                               'PhysicalSizeX': 0.001, 'PhysicalSizeXUnit': 'mm',
                               'PhysicalSizeY': 0.001, 'PhysicalSizeYUnit': 'mm'})
    fast = read_tiff_source_metadata(path)
    compare(fast, reference_metadata(path))
    assert fast['pixel_sizes'][0] == pytest.approx({'y': 1.0, 'x': 1.0})


def test_ome_plane_position_is_read(tmp_path):
    path = str(tmp_path / 'pos.ome.tiff')
    tifffile.imwrite(path, np.zeros((256, 256), dtype=np.uint16),
                     metadata={'axes': 'YX',
                               'PhysicalSizeX': 1.0, 'PhysicalSizeY': 1.0,
                               'Plane': {'PositionX': [12.0], 'PositionXUnit': ['µm'],
                                         'PositionY': [-4.0], 'PositionYUnit': ['µm']}})
    assert read_tiff_source_metadata(path)['position'] == pytest.approx({'x': 12.0, 'y': -4.0})
    assert create_image_source(path).position == pytest.approx({'x': 12.0, 'y': -4.0})


def test_pyramidal_ome_levels_scale_pixel_size(tmp_path):
    path = write_pyramid(tmp_path / 'pyr.ome.tiff', (2048, 2048), levels=3,
                         metadata={'axes': 'YX', 'PhysicalSizeX': 0.5, 'PhysicalSizeXUnit': 'µm',
                                   'PhysicalSizeY': 0.5, 'PhysicalSizeYUnit': 'µm'})
    fast = read_tiff_source_metadata(path)
    compare(fast, reference_metadata(path))
    assert [size['x'] for size in fast['pixel_sizes']] == pytest.approx([0.5, 1.0, 2.0])


def test_source_builds_no_arrays_at_init_but_still_loads_them(tmp_path):
    path = write_pyramid(tmp_path / 'pyr.tiff', (2048, 2048))
    source = create_image_source(path)

    assert source._data_loaded is False
    assert source.shapes == [tuple(shape) for shape in reference_metadata(path)['shapes']]

    data = source.data
    assert source._data_loaded is True
    assert [tuple(level.shape) for level in data] == [tuple(s) for s in source.shapes]
    assert np.asarray(data[-1]).sum() == 0


def test_source_matches_the_reference_end_to_end(tmp_path):
    path = write_pyramid(tmp_path / 'pyr.ome.tiff', (2048, 2048), levels=3,
                         metadata={'axes': 'YX', 'PhysicalSizeX': 0.25,
                                   'PhysicalSizeXUnit': 'µm', 'PhysicalSizeY': 0.25,
                                   'PhysicalSizeYUnit': 'µm'})
    source = create_image_source(path)
    reference = reference_metadata(path)

    assert source.dimension_order == reference['dimension_order']
    assert [tuple(s) for s in source.shapes] == reference['shapes']
    assert source.dtype == reference['dtype']
    assert source.channels == reference['channels']
    for got, want in zip(source.pixel_sizes, reference['pixel_sizes']):
        for dim in want:
            assert got[dim] == pytest.approx(want[dim])


def test_multi_image_ome_xml_parsing_stops_once_settled():
    """The reason this exists: a multi-file OME-TIFF carries the whole dataset's XML per file,
    so the parse must not scale with it. Proven structurally rather than by the clock (a
    wall-time bound is flaky under load): the document is malformed well past the first fed
    chunk, so reaching the end would raise."""
    def xml(count, tail=''):
        images = ''.join(
            f'<Image ID="Image:{i}"><Pixels ID="Pixels:{i}" Type="uint16" SizeX="64" SizeY="64"'
            f' SizeC="1" SizeZ="1" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
            f'<Channel ID="Channel:{i}:0" Name="ch{i}"/>'
            f'<Plane TheC="0" TheZ="0" TheT="0" PositionX="{i}.0" PositionY="0.0"/>'
            f'</Pixels></Image>' for i in range(count))
        return ('<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/'
                f'2016-06">{images}{tail}</OME>')

    small = xml(2)
    large = xml(4000, tail='<Unclosed>' * 5 + '<<<not xml&&&')
    assert len(large) > 512 * 1024

    # scale and channels come from the first Image either way; position is voided for
    # multi-Image - and the malformed tail is never reached, so nothing raises
    for document in (small, large):
        metadata = extract_ome_image_metadata(document)
        assert metadata['scale'] == {'x': 0.5, 'y': 0.5}
        assert metadata['channel_names'] == ['ch0']
        assert metadata['position'] == {}
