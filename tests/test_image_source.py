import numpy as np
import pytest
from pathlib import Path

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.TiffImageSource import TiffImageSource
from muvis_align.image.ZarrImageSource import ZarrImageSource
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import combine_transforms
from muvis_align.util import create_transform, find_all_numbers, split_numeric_dict

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'S000'

TIFF_FILES = [
    '000_000_0.tiff',
    '000_001_0.tiff',
    '001_000_0.tiff',
    '001_001_0.tiff',
]

ZARR_FILES = [
    'S000_000_000.ome.zarr',
    'S000_000_001.ome.zarr',
    'S000_001_000.ome.zarr',
    'S000_001_001.ome.zarr',
]


def _expected_position(filename, formula):
    filename_numeric = find_all_numbers(str(filename))
    context = {'fn': filename_numeric}
    return {dim: eval(expr, context) for dim, expr in formula.items()}


def _sim_at(source, level=0):
    # source.msim is the public, native-dimension_order attribute; source.get_msim(output_order)
    # is a separate, cached, redimensioned-on-demand view used by build_source_msim() - this just
    # extracts one scale's sim straight off source.msim
    return msi_utils.get_sim_from_msim(source.msim, scale=f'scale{level}')


@pytest.mark.parametrize('filename', TIFF_FILES)
def test_tiff_image_source_labels_forced_c_dim_even_when_single_channel(filename):
    """A single-channel source has no 'c' in its native dimension_order, so
    si_utils.get_sim_from_array forces one anyway (size 1) - it must still be labeled with
    source.get_channels()'s channel name (e.g. 'channel 0'), not left as a plain integer index,
    since registration's 'channel' param selects by that label via .sel(c=...)."""
    source = TiffImageSource(str(DATA_DIR / filename))
    assert 'c' not in source.dimension_order

    channel_label = source.get_channels()[0]['label']
    sim0 = _sim_at(source, 0)
    assert list(sim0.coords['c'].values) == [channel_label]

    # the exact call registration/preview_registration makes to select a channel by name
    selected = msi_utils.multiscale_sel_coords(source.get_msim('yx'), {'c': channel_label})
    assert 'c' not in msi_utils.get_sim_from_msim(selected, scale='scale0').dims


@pytest.mark.parametrize('filename', TIFF_FILES)
def test_tiff_image_source_basic(filename):
    source = TiffImageSource(str(DATA_DIR / filename))

    assert source.dimension_order == 'yx'
    assert source.shape == source.shapes[0]
    assert source.dtype is not None

    assert msi_utils.is_msim(source.msim)

    sim0 = _sim_at(source, 0)
    assert (sim0.sizes['y'], sim0.sizes['x']) == tuple(source.shape)


@pytest.mark.parametrize('filename', TIFF_FILES)
def test_tiff_image_source_metadata_overrides_reach_msim(filename):
    source_metadata = {
        'scale': {'x': 0.004, 'y': 0.004},
        'position': {'z': 'fn[-4]', 'y': 'fn[-3]*24', 'x': 'fn[-2]*24'},
    }
    source = TiffImageSource(str(DATA_DIR / filename), source_metadata=source_metadata)

    expected_position = _expected_position(
        DATA_DIR / filename,
        {'z': 'fn[-4]', 'y': 'fn[-3]*24', 'x': 'fn[-2]*24'},
    )

    pixel_size = source.get_pixel_size()
    assert pixel_size['x'] == pytest.approx(0.004)
    assert pixel_size['y'] == pytest.approx(0.004)

    position = source.get_position()
    assert position['x'] == pytest.approx(expected_position['x'])
    assert position['y'] == pytest.approx(expected_position['y'])
    assert position['z'] == pytest.approx(expected_position['z'])

    # the override must reach the msim's own geometry, not just the plain getters
    sim0 = _sim_at(source, 0)
    spacing = si_utils.get_spacing_from_sim(sim0)
    origin = si_utils.get_origin_from_sim(sim0)
    assert spacing['x'] == pytest.approx(0.004)
    assert spacing['y'] == pytest.approx(0.004)
    assert origin['x'] == pytest.approx(expected_position['x'])
    assert origin['y'] == pytest.approx(expected_position['y'])


@pytest.mark.parametrize('filename', ZARR_FILES)
def test_zarr_image_source_basic(filename):
    source = ZarrImageSource(str(DATA_DIR / filename))

    scale_keys = msi_utils.get_sorted_scale_keys(source.msim)
    assert len(scale_keys) == 3  # real 0/1/2 resolution levels on disk

    sim0 = _sim_at(source, 0)
    sim1 = _sim_at(source, 1)
    assert sim1.sizes['x'] == sim0.sizes['x'] // 2
    assert sim1.sizes['y'] == sim0.sizes['y'] // 2


@pytest.mark.parametrize('output_order,z_scale', [('yx', None), ('zyx', 2.5)])
def test_build_source_shape_sim_matches_build_source_msim_scale0(output_order, z_scale):
    """build_source_shape_sim() (the cheap, source.msim-independent geometry used for the
    shapes/overlap-shapes preview) must produce output identical to today's real path
    (build_source_msim()'s scale0, derived from source.msim) - including the z-padding
    convention applied when output_order forces a dim ('z') a 2D source doesn't natively have,
    the specific edge case flagged as the main correctness risk of this shortcut."""
    from muvis_align.image.util import build_source_msim, build_source_shape_sim, \
        create_image_shapes, create_overlap_shapes
    from muvis_align.util import create_transform

    transform_key = 'source_metadata'
    sources = [TiffImageSource(str(DATA_DIR / f)) for f in TIFF_FILES[:2]]
    translations = [{'x': 0.0, 'y': 0.0}, {'x': 50.0, 'y': 30.0}]
    rotations = [0, 15]
    matrix_size = len([dim for dim in output_order if dim in 'xyz']) + 1

    real_sims, cheap_sims = [], []
    for source, translation, rotation in zip(sources, translations, rotations):
        transform = param_utils.invert_coordinate_order(
            create_transform(translation, rotation, matrix_size=matrix_size))
        real_msim = build_source_msim(source, output_order, translation, transform, transform_key,
                                      z_scale=z_scale)
        real_sims.append(msi_utils.get_sim_from_msim(real_msim, scale='scale0'))
        cheap_sims.append(build_source_shape_sim(source, output_order, translation, transform,
                                                  transform_key, z_scale=z_scale))

    for real_sim, cheap_sim in zip(real_sims, cheap_sims):
        real_props = si_utils.get_stack_properties_from_sim(real_sim, transform_key=transform_key)
        cheap_props = si_utils.get_stack_properties_from_sim(cheap_sim, transform_key=transform_key)
        for key in ('shape', 'spacing', 'origin'):
            assert real_props[key].keys() == cheap_props[key].keys()
            for dim in real_props[key]:
                assert real_props[key][dim] == pytest.approx(cheap_props[key][dim])
        np.testing.assert_allclose(np.asarray(real_props['transform']), np.asarray(cheap_props['transform']))

    real_shapes = create_image_shapes(real_sims, transform_key=transform_key)
    cheap_shapes = create_image_shapes(cheap_sims, transform_key=transform_key)
    assert len(real_shapes) == len(cheap_shapes)
    for real_shape, cheap_shape in zip(real_shapes, cheap_shapes):
        np.testing.assert_allclose(real_shape, cheap_shape)

    real_overlap_shapes, real_pairs = create_overlap_shapes(real_sims, transform_key=transform_key)
    cheap_overlap_shapes, cheap_pairs = create_overlap_shapes(cheap_sims, transform_key=transform_key)
    assert [tuple(pair) for pair in real_pairs] == [tuple(pair) for pair in cheap_pairs]
    for real_shape, cheap_shape in zip(real_overlap_shapes, cheap_overlap_shapes):
        np.testing.assert_allclose(real_shape, cheap_shape)


def test_build_source_shape_sim_promote_z_matches_make_msims_3d():
    """When every source is individually 2D (output_order has no 'z') but different sources sit
    at different z heights - e.g. a z-stack of 2D tiles - each source's own z position must still
    reach the shapes/overlap-shapes preview as a real 'z' coordinate, not be silently dropped.
    The production (self.view_msims) path gets this via make_msims_3d(); build_source_shape_sim()
    must produce the same result via its own promote_z=True, without ever building a real msim."""
    from muvis_align.image.util import build_source_msim, build_source_shape_sim, make_msims_3d, \
        create_image_shapes, create_overlap_shapes
    from muvis_align.util import create_transform

    transform_key = 'source_metadata'
    output_order = 'yx'  # every source is natively 2D - output_order itself carries no 'z'
    sources = [TiffImageSource(str(DATA_DIR / f)) for f in TIFF_FILES[:2]]
    translations = [{'x': 0.0, 'y': 0.0, 'z': 0.0}, {'x': 50.0, 'y': 30.0, 'z': 10.0}]
    rotations = [0, 0]

    real_sims, cheap_sims = [], []
    for index, (source, translation, rotation) in enumerate(zip(sources, translations, rotations)):
        transform = param_utils.invert_coordinate_order(
            create_transform(translation, rotation, matrix_size=3))
        real_msim = build_source_msim(source, output_order, translation, transform, transform_key)
        promoted_msim = make_msims_3d([real_msim], positions=[translation])[0]
        real_sims.append(msi_utils.get_sim_from_msim(promoted_msim, scale='scale0'))
        cheap_sims.append(build_source_shape_sim(source, output_order, translation, transform,
                                                  transform_key, promote_z=True))

    for sim, expected_z in zip(real_sims, [0.0, 10.0]):
        assert 'z' in sim.dims
        assert si_utils.get_origin_from_sim(sim)['z'] == pytest.approx(expected_z)
    for sim, expected_z in zip(cheap_sims, [0.0, 10.0]):
        assert 'z' in sim.dims
        assert si_utils.get_origin_from_sim(sim)['z'] == pytest.approx(expected_z)

    real_shapes = create_image_shapes(real_sims, transform_key=transform_key)
    cheap_shapes = create_image_shapes(cheap_sims, transform_key=transform_key)
    assert len(real_shapes) == len(cheap_shapes) == 2
    for real_shape, cheap_shape in zip(real_shapes, cheap_shapes):
        np.testing.assert_allclose(real_shape, cheap_shape)
        # a promoted 2D tile's box must show its own real z, not a dropped/default 0
        assert np.asarray(real_shape).shape[1] == 3

    real_overlap_shapes, real_pairs = create_overlap_shapes(real_sims, transform_key=transform_key)
    cheap_overlap_shapes, cheap_pairs = create_overlap_shapes(cheap_sims, transform_key=transform_key)
    assert [tuple(pair) for pair in real_pairs] == [tuple(pair) for pair in cheap_pairs]
    for real_shape, cheap_shape in zip(real_overlap_shapes, cheap_overlap_shapes):
        np.testing.assert_allclose(real_shape, cheap_shape)


@pytest.mark.parametrize('filename', ZARR_FILES)
def test_zarr_image_source_never_extracts_sims_or_populates_data(filename):
    """ZarrImageSource never calls msi_utils.get_sim_from_msim (metadata comes straight from each
    scale's own 'image' DataArray), and self.data holds one raw dask array per level, opened
    directly off the store - which is what lets the base class synthesize coarse levels for a
    single-resolution store (see ZarrImageSource._load_data)."""
    source = ZarrImageSource(str(DATA_DIR / filename))

    assert len(source.data) == len(source.shapes)
    assert [tuple(data.shape) for data in source.data] == [tuple(shape) for shape in source.shapes]
    for level in range(len(source.pixel_sizes)):
        level_data = source.get_level_data(level)
        sim_data = _sim_at(source, level).data
        assert level_data.shape == sim_data.shape
        np.testing.assert_array_equal(np.asarray(level_data.compute()), np.asarray(sim_data.compute()))


def test_zarr_image_source_scale_override_reaches_getters_and_msim_coords():
    """A source_metadata scale/position override reaches get_pixel_size()/get_position() - what
    the real registration pipeline actually reads - and, since ZarrImageSource now builds its
    msim from raw per-level arrays the same way TiffImageSource does, the msim's own coordinates
    too. (It previously kept read_msim_from_ome_zarr's native coordinates instead, so the two
    formats disagreed on a point nothing downstream reads: build_source_msim() re-assigns every
    level's coordinates from source.pixel_sizes regardless.)"""
    native_pixel_size = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0])).get_pixel_size()
    assert native_pixel_size['x'] != pytest.approx(0.01)

    source_metadata = {'scale': {'x': 0.01, 'y': 0.01}, 'position': {'x': 5, 'y': 7}}
    source = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0]), source_metadata=source_metadata)

    # the override is reflected in the plain getters
    assert source.get_pixel_size()['x'] == pytest.approx(0.01)
    assert source.get_position()['x'] == pytest.approx(5)

    # ...and in the msim's own coordinates, level by level
    sim0 = _sim_at(source, 0)
    sim1 = _sim_at(source, 1)
    assert si_utils.get_spacing_from_sim(sim0)['x'] == pytest.approx(0.01)
    assert si_utils.get_spacing_from_sim(sim1)['x'] == pytest.approx(0.02)  # level 1 is half the resolution
    assert si_utils.get_origin_from_sim(sim0)['x'] == pytest.approx(5)


def test_zarr_image_source_restamped_affine_matches_native_shape():
    """ZarrImageSource keeps the msim read_msim_from_ome_zarr() builds natively (2D data still
    gets a 4x4 identity transform there, since z is a real if trivial spatial dim in NGFF) and
    replaces just that transform with source.transform, in place. The replacement must end up
    the same shape/convention si_utils.get_sim_from_array uses (2D x_in/x_out, no 't' dim) - not
    left as a stale 4x4 with NaN from a shape mismatch against the dropped native transform."""
    source = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0]), source_metadata={'rotation': 30})

    own = param_utils.invert_coordinate_order(
        create_transform(source.position, source.rotation, matrix_size=3))

    scale_keys = msi_utils.get_sorted_scale_keys(source.msim)
    assert len(scale_keys) == 3
    for scale_key in scale_keys:
        sim = msi_utils.get_sim_from_msim(source.msim, scale=scale_key)
        affine = si_utils.get_affine_from_sim(sim, source.transform_key)
        assert affine.shape == (3, 3)
        assert not np.isnan(affine.values).any()
        np.testing.assert_allclose(affine.values, own)


def test_get_msim_caches_by_output_order():
    """get_msim(output_order) redimensions self.msim once per distinct output_order and reuses
    the cached result on repeat calls, rather than redoing the redimension every time
    build_source_msim() is called (e.g. on every init_data() re-run)."""
    source = TiffImageSource(str(DATA_DIR / TIFF_FILES[0]))

    msim_1 = source.get_msim('yx')
    msim_2 = source.get_msim('yx')

    assert msim_1 is msim_2
    assert list(source._redimensioned_msims.keys()) == ['yx']

    image0 = msi_utils.get_sim_from_msim(msim_1, scale='scale0')
    assert image0.dims == ('t', 'c', 'y', 'x')
    assert (image0.sizes['y'], image0.sizes['x']) == tuple(source.shape)


def test_create_image_source_dispatches_on_extension():
    tiff_source = create_image_source(str(DATA_DIR / TIFF_FILES[0]))
    zarr_source = create_image_source(str(DATA_DIR / ZARR_FILES[0]))
    assert isinstance(tiff_source, TiffImageSource)
    assert isinstance(zarr_source, ZarrImageSource)


def test_extra_metadata_composes_with_own_rotation_transform():
    extra_transform = np.eye(3)
    extra_transform[0, 2] = 100  # translate x by 100

    source = TiffImageSource(
        str(DATA_DIR / TIFF_FILES[0]),
        source_metadata={'rotation': 15},
        extra_metadata={'t1': extra_transform.tolist()},
        file_label='t1',
    )

    own = param_utils.invert_coordinate_order(
        create_transform(source.position, source.rotation, matrix_size=3))
    expected = np.array(combine_transforms([own, extra_transform]))

    assert source.transform is not None
    np.testing.assert_allclose(source.transform, expected)

    # reads the transform straight off the msim - no need to extract a sim just for this
    affine = msi_utils.get_transform_from_msim(source.msim, source.transform_key)
    np.testing.assert_allclose(np.array(affine), expected)


def test_build_source_shape_sim_opens_no_source_data():
    """Shape/overlap geometry must stay metadata-only.

    Nothing downstream reads a pixel off these sims, so reaching for source.data (or
    get_level_data(), which for OME-Zarr builds the whole msim) would open every source file
    just to compute bounding boxes - and serially, on the GUI thread, undoing the parallel
    deferral in ImageSource.data. That regression cost 7.5 minutes for 4733 TIFF sources and
    11.4 for OME-Zarr, so the invariant is asserted rather than left to a comment.
    """
    from muvis_align.image.util import build_source_shape_sim

    sources = [TiffImageSource(str(DATA_DIR / f)) for f in TIFF_FILES[:2]]
    for source in sources:
        assert source._data_loaded is False

    sims = [build_source_shape_sim(source, 'tcyx', {'x': 0.0, 'y': 0.0}, None, 'source_metadata')
            for source in sources]

    for source, sim in zip(sources, sims):
        assert source._data_loaded is False, 'shape geometry must not load the source arrays'
        assert source._msim is None, 'shape geometry must not build the source msim'
        # and it still describes the real image
        assert sim.sizes['y'] == source.get_shape(0)[source.dimension_order.index('y')]
        assert sim.sizes['x'] == source.get_shape(0)[source.dimension_order.index('x')]
        assert sim.dtype == source.dtype
