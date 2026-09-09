"""Promoting 2D slices to a size-1 z must happen once, and must not happen twice.

The z dim is essential: it is what lets napari step through a stack of 2D sections one plane at
a time. Nothing here removes it - the only thing removed is doing the work a second time on a
msim that already carries it, which the ordinary preview path did (the viewer promotes each
source's msim, takes a preview sub-pyramid from the result, and hands that to fuse(), which
promoted again because the sources still sit at several z positions).
"""
import glob
import os
from pathlib import Path

import numpy as np
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import param_utils

from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import (build_source_msim, make_msims_3d, msim_is_already_3d,
                                    widen_xaffine_to_3d)
from muvis_align.util import create_transform

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'S000'
KEY = 'source_metadata'


def sources_and_positions(count=3):
    files = sorted(str(path) for path in DATA_DIR.glob('*.tiff'))[:count]
    sources = [create_image_source(name) for name in files]
    positions = [{'z': float(index), 'y': 10.0 * index, 'x': 20.0 * index}
                 for index in range(len(sources))]
    return sources, positions


def geometry(msim):
    out = []
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        image = msim[scale_key].ds['image']
        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        out.append((tuple(image.shape),
                    dict(si_utils.get_spacing_from_sim(image)),
                    dict(si_utils.get_origin_from_sim(image)),
                    np.asarray(si_utils.get_affine_from_sim(sim, transform_key=KEY))))
    return out


def assert_same_geometry(a, b):
    ga, gb = geometry(a), geometry(b)
    assert len(ga) == len(gb)
    for level, (x, y) in enumerate(zip(ga, gb)):
        assert x[0] == y[0], f'level {level} shape'
        assert x[1] == y[1], f'level {level} spacing'
        assert x[2] == y[2], f'level {level} origin'
        np.testing.assert_allclose(x[3], y[3], err_msg=f'level {level} affine')


def test_promotion_still_adds_the_z_dim_napari_steps_through():
    sources, positions = sources_and_positions()
    flat = [build_source_msim(source, 'tcyx', position, None, KEY)
            for source, position in zip(sources, positions)]
    assert all('z' not in m['scale0'].ds['image'].dims for m in flat)

    promoted = make_msims_3d(flat, z_scale=1.0, positions=positions)

    for msim, position in zip(promoted, positions):
        for scale_key in msi_utils.get_sorted_scale_keys(msim):
            image = msim[scale_key].ds['image']
            assert 'z' in image.dims, 'the z dim is what napari slices through - never drop it'
            assert image.sizes['z'] == 1
            # and at this slice's own height, so the sections stack in the right order
            assert float(image.coords['z'].values[0]) == pytest.approx(position['z'])


def test_a_flat_msim_is_not_mistaken_for_a_promoted_one():
    sources, positions = sources_and_positions()
    flat = [build_source_msim(source, 'tcyx', position, None, KEY)
            for source, position in zip(sources, positions)]

    assert not any(msim_is_already_3d(msim) for msim in flat)


def test_a_promoted_msim_is_recognised_and_left_alone():
    sources, positions = sources_and_positions()
    flat = [build_source_msim(source, 'tcyx', position, None, KEY)
            for source, position in zip(sources, positions)]
    promoted = make_msims_3d(flat, z_scale=1.0, positions=positions)

    assert all(msim_is_already_3d(msim) for msim in promoted)

    again = make_msims_3d(promoted, z_scale=1.0, positions=positions)

    # returned untouched, not rebuilt into an equal-but-new tree
    for before, after in zip(promoted, again):
        assert after is before


def test_promoting_twice_changes_nothing():
    sources, positions = sources_and_positions()
    flat = [build_source_msim(source, 'tcyx', position, None, KEY)
            for source, position in zip(sources, positions)]
    once = make_msims_3d(flat, z_scale=1.0, positions=positions)
    twice = make_msims_3d(once, z_scale=1.0, positions=positions)

    for a, b in zip(once, twice):
        assert_same_geometry(a, b)


def test_a_natively_3d_source_is_left_alone_too():
    # a real volume already has z; promotion must not add a second one or rebuild it
    import xarray as xr
    from multiview_stitcher import msi_utils as mu

    sim = si_utils.get_sim_from_array(np.zeros((4, 32, 32), dtype=np.uint16), dims=['z', 'y', 'x'],
                                      scale={'z': 2.0, 'y': 1.0, 'x': 1.0},
                                      translation={'z': 0.0, 'y': 0.0, 'x': 0.0},
                                      transform_key=KEY)
    msim = mu.get_msim_from_sim(sim, scale_factors=[])
    assert msim_is_already_3d(msim)

    result = make_msims_3d([msim], z_scale=1.0, positions=[{'z': 0.0}])

    assert result[0] is msim
    assert result[0]['scale0'].ds['image'].sizes['z'] == 4


# --- the transform widening the promotion depends on ---

def reference_widen(transform):
    """The label-based implementation the numpy one replaces."""
    if 4 in transform.shape:
        return transform
    transform_3d = param_utils.identity_transform(ndim=3)
    if 't' in transform.dims:
        transform = transform.sel(t=0)
    transform_3d.loc[{dim: transform.coords[dim] for dim in transform.dims}] = transform
    return transform_3d


@pytest.mark.parametrize('label, transform', [
    ('identity', param_utils.identity_transform(ndim=2)),
    ('translation', param_utils.affine_to_xaffine(
        create_transform({'x': 5.0, 'y': 7.0}, 0, matrix_size=3))),
    ('rotation + translation', param_utils.affine_to_xaffine(
        create_transform({'x': -3.5, 'y': 11.25}, 37, matrix_size=3))),
])
def test_widening_matches_the_label_based_original(label, transform):
    np.testing.assert_allclose(np.asarray(widen_xaffine_to_3d(transform)),
                               np.asarray(reference_widen(transform)))


def test_widening_leaves_an_already_3d_transform_untouched():
    transform = param_utils.identity_transform(ndim=3)
    assert widen_xaffine_to_3d(transform) is transform
