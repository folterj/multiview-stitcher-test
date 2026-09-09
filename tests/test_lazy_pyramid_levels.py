"""Synthesized coarse levels: metadata at init, arrays only when the data is asked for.

A single-resolution source needs coarser levels so napari has something small to draw while
zoomed out. Deciding *how many and how coarse* is pure shape arithmetic, so it settles during
__init__ - otherwise get_shape/get_pixel_size/get_level_from_scale would answer differently
before and after something first touched the pixel data. Building the arrays is the part that
waits for self.data.

Both halves come from one rule (util.calc_pyramid_level_factors), shared with the OME-Zarr
export side so a written store carries the levels a reader would otherwise have to invent.
"""
import numpy as np
import pytest
import tifffile

from muvis_align.constants import default_chunk_size
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import build_missing_pyramid_levels, calc_pyramid_level_factors


@pytest.fixture
def flat_tiff(tmp_path):
    """A plain, non-pyramidal TIFF - one real resolution."""
    path = tmp_path / 'flat.tiff'
    tifffile.imwrite(str(path), np.zeros((4096, 3072), dtype=np.uint16))
    return str(path)


def test_level_metadata_is_complete_before_any_data_access(flat_tiff):
    source = create_image_source(flat_tiff)

    # no array work has happened yet
    assert source._data_loaded is False
    # but every level is already described
    assert source.shapes == [(4096, 3072), (2048, 1536), (1024, 768)]
    assert [size['x'] for size in source.pixel_sizes] == [1.0, 2.0, 4.0]
    assert [factor['x'] for factor in source.scale_factors] == [1.0, 2.0, 4.0]
    assert max(source.shapes[-1]) <= default_chunk_size


def test_arrays_appear_only_on_data_access_and_match_the_metadata(flat_tiff):
    source = create_image_source(flat_tiff)
    assert source._data_loaded is False

    data = source.data

    assert source._data_loaded is True
    assert [tuple(level.shape) for level in data] == [tuple(s) for s in source.shapes]


def test_msim_access_also_triggers_the_arrays(flat_tiff):
    source = create_image_source(flat_tiff)
    assert source._data_loaded is False

    msim = source.msim

    assert source._data_loaded is True
    from multiview_stitcher import msi_utils
    keys = msi_utils.get_sorted_scale_keys(msim)
    assert len(keys) == len(source.shapes)


def test_repeated_access_does_not_keep_adding_levels(flat_tiff):
    source = create_image_source(flat_tiff)
    first = [tuple(level.shape) for level in source.data]
    for _ in range(3):
        assert [tuple(level.shape) for level in source.data] == first
    assert source.shapes == [tuple(shape) for shape in source.shapes]
    assert len(source.data) == len(source.shapes)


def test_an_already_pyramidal_source_synthesizes_nothing(tmp_path):
    path = tmp_path / 'pyramid.tiff'
    data = np.zeros((2048, 2048), dtype=np.uint16)
    with tifffile.TiffWriter(str(path)) as writer:
        writer.write(data, subifds=1, tile=(256, 256))
        writer.write(data[::2, ::2], subfiletype=1, tile=(256, 256))

    source = create_image_source(str(path))
    assert source._synthesized_level_factors() == []
    assert len(source.shapes) == 2
    assert len(source.data) == 2


def test_zarr_with_its_own_pyramid_synthesizes_nothing(tmp_path):
    from muvis_align.image.ome_zarr_helper import save_ome_multiscale_levels

    path = str(tmp_path / 'store.ome.zarr')
    save_ome_multiscale_levels(path, [(np.zeros((4096, 3072), dtype=np.uint16),
                                       {'y': 1.0, 'x': 1.0})], 'yx', [], {'y': 0.0, 'x': 0.0})
    source = create_image_source(path)

    assert source._synthesized_level_factors() == []
    # its coarse levels come from the file itself (written by the export side's shared rule)
    assert len(source.shapes) > 1
    assert len(source.data) == len(source.shapes)


def test_single_resolution_zarr_synthesizes_levels(tmp_path):
    """A store with no pyramid of its own gets one synthesized, exactly as a flat TIFF does.

    Unlike a decoded TIFF page this is not free - strided subsampling of a chunked store re-reads
    every chunk it touches - but without it nothing downstream can reduce resolution at all, so a
    coarse preview ends up fusing at the store's native resolution: 64x the output pixels for an
    8x preview.
    """
    import zarr

    path = str(tmp_path / 'flat.ome.zarr')
    root = zarr.open_group(path, mode='w', zarr_format=3)
    root.create_array('scale0/image', shape=(4096, 3072), chunks=(512, 512), dtype='uint16')
    root.attrs['ome'] = {
        'version': '0.5',
        'multiscales': [{
            'axes': [{'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                     {'name': 'x', 'type': 'space', 'unit': 'micrometer'}],
            'datasets': [{'path': 'scale0/image', 'coordinateTransformations': [
                {'type': 'scale', 'scale': [1.0, 1.0]},
                {'type': 'translation', 'translation': [0.0, 0.0]}]}],
        }],
    }
    zarr.consolidate_metadata(root.store)

    source = create_image_source(path)

    # the store itself has exactly one dataset - every further level here is synthesized
    assert len(source.shapes) > 1
    assert [tuple(data.shape) for data in source.data] == [tuple(shape) for shape in source.shapes]
    # coarse enough to draw a zoomed-out overview from, per the shared rule
    assert max(source.get_shape(len(source.shapes) - 1)) <= max(source.get_shape(0)) // 2


# --- the shared rule itself ---

def test_factors_halve_until_small_enough():
    factors = calc_pyramid_level_factors({'y': 4096, 'x': 4096})
    assert [f['x'] for f in factors] == [2, 4]


def test_factors_are_cumulative_not_per_step():
    factors = calc_pyramid_level_factors({'y': 8192, 'x': 8192})
    assert [f['y'] for f in factors] == [2, 4, 8]


def test_no_factors_when_already_small_enough():
    assert calc_pyramid_level_factors({'y': 512, 'x': 512}) == []


def test_anisotropic_extents_keep_halving_only_while_they_can():
    # y/x drive the stopping rule; z bottoms out at 1 rather than going fractional
    factors = calc_pyramid_level_factors({'z': 3, 'y': 4096, 'x': 4096})
    assert [f['x'] for f in factors] == [2, 4]
    assert [int(np.ceil(3 / f['z'])) for f in factors] == [2, 1]


def test_no_spatial_dims_is_a_no_op():
    assert calc_pyramid_level_factors({}) == []


def test_odd_extents_round_up_like_strided_slicing():
    # data[::2] on 4097 rows yields 2049, not 2048 - the factors must be derived the same way,
    # or the committed shapes would not match the arrays built later
    factors = calc_pyramid_level_factors({'y': 4097, 'x': 4097})
    for level_factors in factors:
        strided = np.zeros((4097, 4097))[::level_factors['y'], ::level_factors['x']]
        assert strided.shape == (int(np.ceil(4097 / level_factors['y'])),
                                 int(np.ceil(4097 / level_factors['x'])))


@pytest.mark.parametrize('shape, dim_order', [
    ((4096, 3072), 'yx'),
    ((4097, 4097), 'yx'),
    ((3, 4096, 4096), 'zyx'),
    ((2, 3, 2048, 2048), 'czyx'),
    ((1000, 1000), 'yx'),
])
def test_committed_metadata_matches_what_the_arrays_actually_become(shape, dim_order):
    """The metadata half runs at init from shapes alone; the array half slices real data later.
    build_missing_pyramid_levels() is the oracle for both - if they disagree, self.shapes and
    self.data would describe different pyramids (which _add_missing_pyramid_level_data asserts
    against, so a drift here is a hard failure at runtime, not a silent one)."""
    import dask.array as da

    data = da.zeros(shape, dtype=np.uint16)
    pixel_size = {dim: 0.5 for dim in dim_order if dim in 'zyx'}
    datas, pixel_sizes = build_missing_pyramid_levels(data, dim_order, pixel_size)

    sizes = {dim: size for dim, size in zip(dim_order, shape) if dim in 'xyz'}
    factors = calc_pyramid_level_factors(sizes)
    assert len(factors) == len(datas) - 1

    axes = {dim: axis for axis, dim in enumerate(dim_order)}
    for level_factors, expected_data, expected_pixel_size in zip(factors, datas[1:], pixel_sizes[1:]):
        committed_shape = tuple(-(-size // level_factors.get(dim, 1))
                                for dim, size in zip(dim_order, shape))
        assert committed_shape == tuple(expected_data.shape)
        committed_pixel_size = {dim: value * shape[axes[dim]] / committed_shape[axes[dim]]
                                for dim, value in pixel_size.items()}
        assert committed_pixel_size == pytest.approx(expected_pixel_size)
