"""A preview must not fuse an unbounded stack to draw a few hundred on-screen pixels.

preview_scale cannot enforce that on its own: it selects a level relative to each source's own
pyramid, so the fused result still grows with the dataset, and the post-pre-processing preview
is built from register_msims and never consults it at all. A run whose pre_processing scale was
1 therefore fused 396.9GB over 55 minutes to show what an 8x-reduced one showed in 9. Bounding
the fused size covers both routes.
"""
import numpy as np
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.util import (drop_finest_msim_level, estimate_fused_size,
                                    reduce_msims_to_fused_size)

KEY = 'affine_metadata'


def make_msim(size, levels, spacing=1.0, origin=(0.0, 0.0)):
    """A pyramid of `levels` halvings, as a source's own msim would be."""
    sims = []
    for level in range(levels):
        factor = 2 ** level
        sim = si_utils.get_sim_from_array(
            np.zeros((size // factor, size // factor), dtype=np.uint16), dims=['y', 'x'],
            scale={'y': spacing * factor, 'x': spacing * factor},
            translation={'y': origin[0], 'x': origin[1]}, transform_key=KEY)
        sims.append(sim)
    return msi_utils.get_msim_from_sims(sims)


def grid(count=2, size=512, levels=4, spacing=1.0):
    return [make_msim(size, levels, spacing, origin=(row * size * spacing * 0.9,
                                                     col * size * spacing * 0.9))
            for row in range(count) for col in range(count)]


def fused_bytes(msims):
    return estimate_fused_size(msims, KEY)[0]


def test_drop_finest_level_removes_one_level():
    msims = grid()
    assert len(msi_utils.get_sorted_scale_keys(msims[0])) == 4

    reduced, changed = drop_finest_msim_level(msims)

    assert changed is True
    assert len(msi_utils.get_sorted_scale_keys(reduced[0])) == 3
    # and the new finest level is the old second one
    assert reduced[0]['scale0'].ds['image'].shape == msims[0]['scale1'].ds['image'].shape


def test_drop_finest_level_is_a_no_op_on_a_single_level_msim():
    msims = grid(levels=1)

    reduced, changed = drop_finest_msim_level(msims)

    assert changed is False
    assert all(a is b for a, b in zip(reduced, msims))


def test_an_estimate_that_already_fits_is_left_untouched():
    msims = grid()
    size = fused_bytes(msims)

    capped = reduce_msims_to_fused_size(msims, KEY, max_bytes=size * 2)

    assert capped is msims


def test_an_oversized_preview_is_reduced_until_it_fits():
    msims = grid()
    size = fused_bytes(msims)
    budget = size // 10

    capped = reduce_msims_to_fused_size(msims, KEY, max_bytes=budget)

    assert fused_bytes(capped) <= budget
    assert fused_bytes(capped) < size
    # reduced by dropping levels, not by discarding sources
    assert len(capped) == len(msims)


def test_single_level_msims_are_returned_without_measuring_them(caplog):
    """Nothing can be dropped, so the estimate could only confirm that at full price (~4s for
    4733 sources). This is the ordinary shape of the preprocessed preview - pre-processing at
    scale 8 has already reduced every source to one level - so it is not worth a warning
    either: that reduction is the setting doing its job."""
    import logging

    msims = grid(levels=1)
    with caplog.at_level(logging.WARNING):
        capped = reduce_msims_to_fused_size(msims, KEY, max_bytes=1)

    assert capped is msims
    assert caplog.text == ''


def test_it_warns_when_it_runs_out_of_levels_part_way(caplog):
    """Started with room to reduce, ran out before fitting - that must end the loop and be
    reported, not spin or silently look successful."""
    import logging

    msims = grid(levels=2)
    with caplog.at_level(logging.WARNING):
        capped = reduce_msims_to_fused_size(msims, KEY, max_bytes=1)

    assert len(msi_utils.get_sorted_scale_keys(capped[0])) == 1, 'should have dropped what it could'
    assert 'no coarser pyramid level remains' in caplog.text
    # reports the size actually being fused, without prescribing a fix that may not apply
    assert 'deeper pyramid' not in caplog.text


def test_reduction_stops_as_soon_as_it_fits_rather_than_going_coarsest():
    msims = grid()
    one_level_down, _ = drop_finest_msim_level(msims)
    budget = fused_bytes(one_level_down)

    capped = reduce_msims_to_fused_size(msims, KEY, max_bytes=budget)

    assert fused_bytes(capped) == budget
    assert len(msi_utils.get_sorted_scale_keys(capped[0])) == 3


def test_estimate_counts_the_output_stack_not_the_sources():
    """The number that matters is the fused result - overlapping tiles do not each add their
    own bytes to it."""
    msims = grid(count=2, size=512, levels=1)
    size = fused_bytes(msims)

    # 4 tiles of 512x512 uint16 laid out with 10% overlap: under 4 separate tiles' worth
    assert size < 4 * 512 * 512 * 2
    assert size > 512 * 512 * 2


@pytest.mark.parametrize('budget_gb', [0.001, 0.01, 0.1])
def test_result_is_always_within_budget_or_as_coarse_as_possible(budget_gb):
    msims = grid(count=3, size=1024, levels=5)
    budget = int(budget_gb * 1024 ** 3)

    capped = reduce_msims_to_fused_size(msims, KEY, max_bytes=budget)

    levels = len(msi_utils.get_sorted_scale_keys(capped[0]))
    assert fused_bytes(capped) <= budget or levels == 1
