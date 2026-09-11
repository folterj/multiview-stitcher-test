"""The on-screen overview is pasted, not fused - see composite_msims_overview()."""
import numpy as np
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import param_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.util import composite_msims_overview, wrap_sims_as_msims

TRANSFORM_KEY = 'affine_metadata'


def make_msim(value, translation, size=8, spacing=1.0, rotate=False):
    array = np.full((1, 1, size, size), value, dtype=np.uint16)
    sim = si_utils.get_sim_from_array(
        array,
        dims=['t', 'c', 'y', 'x'],
        scale={'y': spacing, 'x': spacing},
        translation={'y': 0.0, 'x': 0.0},
        transform_key=TRANSFORM_KEY,
    )
    affine = si_utils.get_affine_from_sim(sim, TRANSFORM_KEY)
    matrix = np.asarray(affine.sel(t=0) if 't' in affine.dims else affine, dtype=float).copy()
    matrix[0, 2] = translation[0]
    matrix[1, 2] = translation[1]
    if rotate:
        # a rotation cannot be pasted - it needs resampling
        matrix[0, 0], matrix[0, 1] = 0.0, -1.0
        matrix[1, 0], matrix[1, 1] = 1.0, 0.0
    si_utils.set_sim_affine(sim, param_utils.affine_to_xaffine(matrix), transform_key=TRANSFORM_KEY)
    return wrap_sims_as_msims([sim])[0]


def overview_array(msim):
    return np.asarray(msi_utils.get_sim_from_msim(msim).data)[0, 0]


def test_composite_places_each_source_at_its_own_position():
    """Two sources side by side land side by side, each keeping its own values."""
    msims = [make_msim(10, (0, 0)), make_msim(20, (0, 8))]

    overview = overview_array(composite_msims_overview(msims, TRANSFORM_KEY))

    assert overview.shape == (8, 16)
    assert np.all(overview[:, :8] == 10)
    assert np.all(overview[:, 8:] == 20)


def test_composite_overwrites_where_sources_overlap():
    """Plain overwrite, deliberately: this is an overview, not the fused result, and the last
    source in is as good a choice as any for a picture of where things sit."""
    msims = [make_msim(10, (0, 0)), make_msim(20, (0, 4))]

    overview = overview_array(composite_msims_overview(msims, TRANSFORM_KEY))

    assert overview.shape == (8, 12)
    assert np.all(overview[:, :4] == 10)      # only the first source
    assert np.all(overview[:, 4:] == 20)      # the second, over the overlap and beyond


def test_composite_keeps_the_geometry_the_fusion_would_have_had():
    msims = [make_msim(10, (0, 0), spacing=2.0), make_msim(20, (0, 16), spacing=2.0)]

    overview = composite_msims_overview(msims, TRANSFORM_KEY)

    sim = msi_utils.get_sim_from_msim(overview)
    assert si_utils.get_spacing_from_sim(sim) == {'y': 2.0, 'x': 2.0}
    assert si_utils.get_origin_from_sim(sim) == {'y': 0.0, 'x': 0.0}


def test_composite_coarsens_rather_than_allocating_more_than_its_budget():
    """The overview is pasted eagerly into one array, so its size is real memory - it is
    coarsened in-plane to fit, not left to allocate whatever the fused geometry implies."""
    msims = [make_msim(10, (0, 0), size=512), make_msim(20, (0, 512), size=512)]

    full = overview_array(composite_msims_overview(msims, TRANSFORM_KEY))
    small = overview_array(composite_msims_overview(msims, TRANSFORM_KEY, max_bytes=64 * 1024))

    assert full.shape == (512, 1024)
    assert small.size * 2 <= 64 * 1024
    assert small.shape[0] < full.shape[0] and small.shape[1] < full.shape[1]
    # ...and it still shows both sources, in the same places
    assert np.all(small[:, :small.shape[1] // 2] == 10)
    assert np.all(small[:, small.shape[1] // 2:] == 20)


def test_composite_declines_a_transform_it_cannot_paste():
    """A rotation needs resampling rather than a paste - the caller falls back to fusing."""
    msims = [make_msim(10, (0, 0)), make_msim(20, (0, 8), rotate=True)]

    assert composite_msims_overview(msims, TRANSFORM_KEY) is None


def test_composite_of_nothing_is_nothing():
    assert composite_msims_overview([], TRANSFORM_KEY) is None
