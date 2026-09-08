"""A written OME-Zarr must carry the coarse levels a reader would otherwise need to invent.

The reader synthesizes missing levels only for sources that can do it cheaply (a non-pyramidal
TIFF page is already decoded); for a chunked store it cannot - strided subsampling of a zarr
array still reads every chunk it touches. So the coarse levels have to be in the file, and the
export must reach the same "small enough to draw a zoomed-out overview from" threshold
(default_chunk_size) that build_missing_pyramid_levels uses on the reader side. ngff_zarr's own
optimal-level-count logic stops one to two levels earlier (it halts below twice the chunk size
and never emits a level smaller than one chunk), which left exactly that gap.
"""
import numpy as np
import pytest

from muvis_align.constants import default_chunk_size
from muvis_align.image.ome_zarr_helper import get_padding_scale_factors, save_ome_multiscale_levels
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import get_level_from_scale


def coarsest_size(shape, dim_order, scale_factors):
    if not scale_factors:
        return max(size for dim, size in zip(dim_order, shape) if dim in 'xyz')
    last = scale_factors[-1]
    return max(size // last.get(dim, 1)
               for dim, size in zip(dim_order, shape) if dim in 'xyz')


@pytest.mark.parametrize('size', [2048, 4096, 6400])
def test_padding_reaches_the_readers_own_threshold(size):
    shape, dim_order = (size, size), 'yx'
    factors = get_padding_scale_factors(shape, dim_order)
    assert coarsest_size(shape, dim_order, factors) <= default_chunk_size


def test_padding_is_cumulative_and_halves_each_level():
    factors = get_padding_scale_factors((4096, 4096), 'yx')
    assert [f['x'] for f in factors] == [2, 4]
    assert [f['y'] for f in factors] == [2, 4]


def test_no_padding_needed_for_an_already_small_level():
    assert get_padding_scale_factors((512, 512), 'yx') == []


def test_ignores_non_spatial_dims():
    factors = get_padding_scale_factors((1, 3, 4096, 4096), 'tcyx')
    assert factors and all(set(f) == {'y', 'x'} for f in factors)


def test_no_spatial_dims_is_a_no_op():
    assert get_padding_scale_factors((5,), 't') == []


def test_written_single_resolution_store_is_usable_at_a_coarse_preview_scale(tmp_path):
    # a source with one real resolution, as an external converter would write it
    data = np.zeros((4096, 4096), dtype=np.uint16)
    path = str(tmp_path / 'single.ome.zarr')
    save_ome_multiscale_levels(path, [(data, {'y': 1.0, 'x': 1.0})], 'yx', [],
                               {'y': 0.0, 'x': 0.0})

    source = create_image_source(path)
    coarsest = max(size for dim, size in zip(source.dimension_order, source.shapes[-1])
                   if dim in 'xyz')
    assert coarsest <= default_chunk_size

    # and the coarse level is genuinely reachable: asking for a 4x reduction gets one, rather
    # than silently falling back to a much finer level (which is what makes a preview explode)
    level, residual, _ = get_level_from_scale(source, 4)
    assert level > 0
    assert max(residual.values()) == 1
