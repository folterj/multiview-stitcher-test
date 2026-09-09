"""get_chunk_sizes() must bound what fusing one output chunk actually costs.

multiview_stitcher's fusion transforms every source overlapping an output chunk into a
full-chunk-sized float32 array and stacks them (fusion._core: field_ims_t, plus a same-shaped
blending-weight stack and their product), so one chunk's peak memory is
~views_in_chunk * chunk_voxels * 4 * fusion_stack_arrays - independent of the output dtype.
fuse() also reuses one output_chunksize for every pyramid level it builds, so the worst case is
a *coarse* level whose whole extent fits in one chunk covering every source.

These tests model that cost over the full pyramid, the way the real fusion graph pays it.
"""
import numpy as np
import pytest
from multiview_stitcher import msi_utils

from muvis_align.constants import fusion_stack_arrays
from muvis_align.image.util import get_chunk_sizes


def worst_chunk_fusion_bytes(chunk_sizes, output_shape, num_sources, num_z_positions):
    """Peak bytes for the most expensive single chunk over every pyramid level fuse() builds."""
    dims = list(output_shape)
    sources_per_plane = max(1, round(num_sources / max(1, num_z_positions)))
    level_shapes, _, _ = msi_utils.calc_resolution_levels(output_shape)
    worst = 0
    for shape in level_shapes:
        # a chunk is clipped to the level's own extent - that clipping is exactly what makes
        # coarse levels span every source
        chunk = {dim: min(chunk_sizes[dim], shape[dim]) for dim in dims}
        xy_fraction = np.prod([chunk[dim] / shape[dim] for dim in dims if dim in ('x', 'y')])
        # sources reaching one chunk: those in the z planes it spans, scaled by its share of
        # the field of view (sources tile the output, so area share ~ source share)
        z_span = chunk.get('z', 1) if num_z_positions > 1 else 1
        views = max(1, min(num_sources, sources_per_plane * z_span) * xy_fraction)
        chunk_voxels = np.prod([chunk[dim] for dim in dims])
        worst = max(worst, views * chunk_voxels * 4 * fusion_stack_arrays)
    return worst


# an explicit budget throughout: the production default (default_fusion_chunk_bytes) is derived
# from the running machine's own CPU/memory allocation, so pinning it here is what keeps these
# assertions the same on a laptop, in CI, and on an HPC node
BUDGET = 256 * 1024 ** 2

# (label, num_sources, num_z_positions, output shape)
CASES = [
    # the case that motivated this: ~4700 2D tiles stacked over 72 sections. Byte-budget-only
    # sizing gave {'z': 32, 'y': 1024, 'x': 1024} here, whose coarse levels each fuse every
    # source in 32 sections at once - over 500 GB for one chunk.
    ('sectioned stack', 4733, 72, {'z': 72, 'y': 6800, 'x': 6800}),
    ('small sectioned stack', 54, 6, {'z': 6, 'y': 2000, 'x': 2000}),
    ('single-plane mosaic', 200, 1, {'y': 20000, 'x': 20000}),
    ('single-plane mosaic 3d', 200, 1, {'z': 8, 'y': 20000, 'x': 20000}),
    ('native z-stacks', 12, 1, {'z': 200, 'y': 4000, 'x': 4000}),
    ('one source', 1, 1, {'z': 100, 'y': 1000, 'x': 1000}),
]


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'float32'])
@pytest.mark.parametrize('label, num_sources, num_z_positions, output_shape', CASES)
def test_chunk_sizes_bound_peak_fusion_memory(label, num_sources, num_z_positions, output_shape,
                                              dtype):
    chunk_sizes = get_chunk_sizes(np.dtype(dtype), list(output_shape),
                                  num_sources=num_sources, num_z_positions=num_z_positions,
                                  fusion_target_bytes=BUDGET)

    assert set(chunk_sizes) == set(output_shape)
    assert all(size >= 1 for size in chunk_sizes.values())

    worst = worst_chunk_fusion_bytes(chunk_sizes, output_shape, num_sources, num_z_positions)
    assert worst <= BUDGET, (f'{label} ({dtype}): chunks {chunk_sizes} need '
                             f'{worst / 1024 ** 2:.0f} MB for one chunk')


def test_sources_spread_over_z_get_single_plane_chunks():
    # a chunk spanning Nz sections costs Nz ** 2 (Nz times the sources, Nz times the voxels),
    # so z must not be widened at all once sources sit at distinct z positions
    chunk_sizes = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'],
                                  num_sources=4733, num_z_positions=72,
                                  fusion_target_bytes=BUDGET)
    assert chunk_sizes['z'] == 1


def test_native_z_stack_still_gets_a_deeper_z_chunk():
    # num_z_positions == 1: every source spans the whole z range, so a deeper z chunk adds
    # voxels but no extra views - the output-byte budget stays the binding constraint there
    chunk_sizes = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'],
                                  num_sources=1, num_z_positions=1,
                                  fusion_target_bytes=BUDGET)
    assert chunk_sizes['z'] > 1
    assert chunk_sizes['y'] == chunk_sizes['x'] == 1024


def test_xy_chunks_shrink_as_sources_multiply():
    def xy(num_sources):
        return get_chunk_sizes(np.dtype('uint16'), ['y', 'x'], num_sources=num_sources,
                               fusion_target_bytes=BUDGET)['x']

    # up to a couple of dozen sources the generous default already fits the budget
    assert xy(1) == xy(10) == 1024
    assert xy(100) < 1024
    assert xy(1000) < xy(100)
    # never below one whole block, however many sources there are
    assert xy(10 ** 6) == 64


def test_a_bigger_budget_buys_bigger_chunks_not_deeper_z():
    """An HPC allocation should spend its headroom on the generous default chunk size (fewer
    chunks, so less graph to build), never on a deeper z chunk - depth is what pulls whole
    extra sections of sources into one chunk."""
    laptop = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                             num_z_positions=72, fusion_target_bytes=64 * 1024 ** 2)
    hpc = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                          num_z_positions=72, fusion_target_bytes=4 * 1024 ** 3)
    assert laptop['z'] == hpc['z'] == 1
    assert laptop['x'] < hpc['x']
    # and never past the generous default, however much headroom there is
    assert hpc['x'] == 1024


def test_default_budget_comes_from_this_machines_allocation():
    from muvis_align.constants import default_fusion_chunk_bytes

    # a real, plausible per-chunk budget - not zero, and not the whole machine
    assert 64 * 1024 ** 2 <= default_fusion_chunk_bytes <= 4 * 1024 ** 3
    # the default path and an explicit equal budget must agree
    assert (get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                            num_z_positions=72)
            == get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                               num_z_positions=72,
                               fusion_target_bytes=default_fusion_chunk_bytes))


def test_preview_scale_shortfall_is_logged(caplog):
    """A source pyramid that cannot reach the requested preview scale silently yields a much
    finer level, and the preview then fuses far more than asked for - so it must be logged."""
    import logging
    from types import SimpleNamespace

    import xarray as xr
    from multiview_stitcher import msi_utils, spatial_image_utils as si_utils

    from muvis_align.image.util import select_msim_subpyramid_at_scale

    def make(levels):
        sim = si_utils.get_sim_from_array(np.zeros((64, 64), dtype=np.uint16), dims=['y', 'x'],
                                          scale={'y': 1.0, 'x': 1.0},
                                          translation={'y': 0.0, 'x': 0.0},
                                          transform_key='affine_metadata')
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        source = SimpleNamespace(
            scale_factors=[{'y': float(f), 'x': float(f)} for f in levels],
            dimension_order='yx',
            get_shape=lambda level=0: (64, 64),
            get_pixel_size=lambda: {'y': 1.0, 'x': 1.0})
        return source, msim

    # only a 2x level available, but 16x asked for -> 8x short
    source, msim = make([1, 2])
    with caplog.at_level(logging.WARNING):
        select_msim_subpyramid_at_scale([msim], [source], 16)
    assert 'Preview scale 16 not reachable' in caplog.text
    assert '8x finer' in caplog.text

    # a pyramid that does reach it logs nothing
    caplog.clear()
    source, msim = make([1, 2, 4, 8, 16])
    with caplog.at_level(logging.WARNING):
        select_msim_subpyramid_at_scale([msim], [source], 16)
    assert 'not reachable' not in caplog.text

    # falling one level short (8 of 16) is not worth crying about
    caplog.clear()
    source, msim = make([1, 2, 4, 8])
    with caplog.at_level(logging.WARNING):
        select_msim_subpyramid_at_scale([msim], [source], 16)
    assert 'not reachable' not in caplog.text


def test_a_size_1_dim_is_not_mistaken_for_a_pyramid_shortfall(caplog):
    """A size-1 dim is identical at every level, so its residual is always the full requested
    factor - counting it reported a 16x shortfall for the perfectly good 4-level pyramids every
    converted OME-Zarr tile has (they carry a size-1 'z')."""
    import logging
    from types import SimpleNamespace

    from multiview_stitcher import msi_utils, spatial_image_utils as si_utils

    from muvis_align.image.util import select_msim_subpyramid_at_scale

    sim = si_utils.get_sim_from_array(np.zeros((1, 64, 64), dtype=np.uint16), dims=['z', 'y', 'x'],
                                      scale={'z': 1.0, 'y': 1.0, 'x': 1.0},
                                      translation={'z': 0.0, 'y': 0.0, 'x': 0.0},
                                      transform_key='affine_metadata')
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    source = SimpleNamespace(
        # x/y really do reach 8x; z cannot be reduced at all
        scale_factors=[{'z': 1.0, 'y': float(f), 'x': float(f)} for f in (1, 2, 4, 8)],
        dimension_order='zyx',
        get_shape=lambda level=0: (1, 64, 64),
        get_pixel_size=lambda: {'z': 1.0, 'y': 1.0, 'x': 1.0})

    with caplog.at_level(logging.WARNING):
        select_msim_subpyramid_at_scale([msim], [source], 16)
    assert 'not reachable' not in caplog.text
