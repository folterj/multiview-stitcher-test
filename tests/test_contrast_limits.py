"""get_contrast_limits() must never be the step that blocks first paint.

It exists to pick display bounds before a layer goes on screen, from the coarsest pyramid level
only. That level is small in pixels but not necessarily in work: fused from thousands of
sources it is thousands of transforms, however few pixels come out - so past a task-count
threshold it falls back to the naive dtype range instead of computing.
"""
import dask.array as da
import numpy as np
import xarray as xr
from multiview_stitcher import msi_utils, spatial_image_utils as si_utils

from muvis_align.image.util import get_contrast_limits


def make_msim(values, chunks=None):
    data = np.asarray(values, dtype=np.uint16)
    sim = si_utils.get_sim_from_array(data, dims=['y', 'x'], scale={'y': 1.0, 'x': 1.0},
                                      translation={'y': 0.0, 'x': 0.0},
                                      transform_key='affine_metadata')
    if chunks is not None:
        sim = sim.chunk(chunks)
    return msi_utils.get_msim_from_sim(sim, scale_factors=[])


def test_computes_real_limits_for_a_cheap_graph():
    msim = make_msim(np.full((8, 8), 700))
    assert get_contrast_limits(msim) == [700.0, 700.0 + 1]


def test_widens_a_flat_range_so_napari_gets_a_usable_span():
    low, high = get_contrast_limits(make_msim(np.zeros((4, 4))))
    assert low < high


def test_cheap_returns_the_dtype_range_without_computing():
    msim = make_msim(np.full((8, 8), 700))
    assert get_contrast_limits(msim, cheap=True) == [0, np.iinfo(np.uint16).max]


def test_falls_back_to_dtype_range_when_the_coarsest_level_is_expensive():
    # many small chunks stands in for the real cost driver (a level fused from thousands of
    # sources): what matters is that the level's own graph is large
    msim = make_msim(np.full((64, 64), 700), chunks={'y': 1, 'x': 1})
    coarsest = msim['scale0'].ds['image'].data
    assert len(coarsest.dask) > 64

    assert get_contrast_limits(msim, max_tasks=64) == [0, np.iinfo(np.uint16).max]
    # and still computes the real range when the same level is under the threshold
    assert get_contrast_limits(msim, max_tasks=10 ** 6) == [700.0, 701.0]


def test_float_data_falls_back_to_unit_range():
    data = np.zeros((4, 4), dtype=np.float32)
    sim = si_utils.get_sim_from_array(data, dims=['y', 'x'], scale={'y': 1.0, 'x': 1.0},
                                      translation={'y': 0.0, 'x': 0.0},
                                      transform_key='affine_metadata')
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    assert get_contrast_limits(msim, cheap=True) == [0.0, 1.0]
