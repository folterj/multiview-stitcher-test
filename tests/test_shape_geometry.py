"""Drawing shapes must not read, create or allocate image data.

Shapes and overlap boxes need geometry - shape, spacing, origin, transform - and nothing else.
Going through a sim to get it meant conjuring an array for the sim to wrap, and (before that)
reading the source's real one, which opens the file. Source init deliberately defers all of
that until something actually processes pixels or builds a preview, so the shapes path must not
undo it.

build_source_stack_props() therefore produces the geometry directly. The sim-based path still
exists for multiview_stitcher's exact overlap test, which takes sims - so the two must agree,
which is what most of this asserts.
"""
from pathlib import Path

import numpy as np
import pytest
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.TiffImageSource import TiffImageSource
from muvis_align.image.util import (build_source_shape_sim, build_source_stack_props,
                                    create_image_shapes, create_overlap_shapes)
from muvis_align.util import create_transform

# same fixtures test_image_source.py uses, restated rather than cross-imported between test
# modules (tests/ is not a package)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'S000'
TIFF_FILES = ['000_000_0.tiff', '000_001_0.tiff', '001_000_0.tiff', '001_001_0.tiff']

TRANSLATION = {'x': 3.0, 'y': 4.0, 'z': 9.0}

# (output_order, z_scale, promote_z, transform matrix size or None)
CASES = [
    ('yx', None, False, None),
    ('yx', None, False, 3),
    ('yx', None, True, None),
    ('yx', None, True, 3),
    ('zyx', 2.5, False, None),
    ('zyx', 2.5, False, 4),
]


def make_transform(matrix_size):
    if matrix_size is None:
        return None
    return create_transform({'x': 5.0, 'y': 7.0}, 10, matrix_size=matrix_size)


@pytest.mark.parametrize('output_order, z_scale, promote_z, matrix_size', CASES)
def test_stack_props_match_the_sim_they_replace(output_order, z_scale, promote_z, matrix_size):
    source = TiffImageSource(str(DATA_DIR / TIFF_FILES[0]))
    transform = make_transform(matrix_size)
    args = (source, output_order, TRANSLATION, transform, 'source_metadata')
    kwargs = dict(z_scale=z_scale, promote_z=promote_z)

    sim = build_source_shape_sim(*args, **kwargs)
    reference = si_utils.get_stack_properties_from_sim(sim, transform_key='source_metadata')
    props = build_source_stack_props(*args, **kwargs)

    assert props['shape'] == reference['shape']
    assert props['spacing'] == reference['spacing']
    assert props['origin'] == reference['origin']
    np.testing.assert_allclose(np.asarray(props['transform']), np.asarray(reference['transform']))


def test_building_geometry_opens_no_file_and_allocates_nothing():
    sources = [TiffImageSource(str(DATA_DIR / name)) for name in TIFF_FILES[:2]]
    for source in sources:
        assert source._data_loaded is False

    props = [build_source_stack_props(source, 'tcyx', {'x': 0.0, 'y': 0.0}, None,
                                      'source_metadata') for source in sources]

    for source in sources:
        assert source._data_loaded is False, 'shape geometry must not load the source arrays'
        assert source._msim is None, 'shape geometry must not build the source msim'
    # plain metadata all the way down - nothing array-shaped is produced at all
    for entry in props:
        assert set(entry) == {'shape', 'spacing', 'origin', 'transform'}
        assert all(isinstance(size, int) for size in entry['shape'].values())


@pytest.mark.parametrize('output_order, z_scale, promote_z, matrix_size', CASES)
def test_image_shapes_identical_from_props_and_from_sims(output_order, z_scale, promote_z,
                                                         matrix_size):
    sources = [TiffImageSource(str(DATA_DIR / name)) for name in TIFF_FILES[:3]]
    translations = [{'x': 0.0, 'y': 0.0}, {'x': 50.0, 'y': 30.0}, {'x': 25.0, 'y': 60.0}]
    transform = make_transform(matrix_size)
    kwargs = dict(z_scale=z_scale, promote_z=promote_z)

    sims = [build_source_shape_sim(source, output_order, translation, transform,
                                   'source_metadata', **kwargs)
            for source, translation in zip(sources, translations)]
    props = [build_source_stack_props(source, output_order, translation, transform,
                                      'source_metadata', **kwargs)
             for source, translation in zip(sources, translations)]

    for force_2d in (False, True):
        from_sims = create_image_shapes(sims, transform_key='source_metadata', force_2d=force_2d)
        from_props = create_image_shapes(props, transform_key='source_metadata', force_2d=force_2d)
        assert len(from_props) == len(from_sims)
        for got, want in zip(from_props, from_sims):
            np.testing.assert_allclose(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize('force_2d', [False, True])
def test_overlap_shapes_identical_from_props_and_from_sims(force_2d):
    """The broad-phase/AABB path - what initial project load uses, with no pairs given."""
    sources = [TiffImageSource(str(DATA_DIR / name)) for name in TIFF_FILES[:3]]
    # deliberately overlapping, so pairs actually survive the broad phase
    translations = [{'x': 0.0, 'y': 0.0}, {'x': 5.0, 'y': 3.0}, {'x': 2.0, 'y': 6.0}]

    sims = [build_source_shape_sim(source, 'yx', translation, None, 'source_metadata')
            for source, translation in zip(sources, translations)]
    props = [build_source_stack_props(source, 'yx', translation, None, 'source_metadata')
             for source, translation in zip(sources, translations)]

    shapes_sims, pairs_sims = create_overlap_shapes(sims, 'source_metadata', force_2d=force_2d)
    shapes_props, pairs_props = create_overlap_shapes(props, 'source_metadata', force_2d=force_2d)

    assert [tuple(pair) for pair in pairs_props] == [tuple(pair) for pair in pairs_sims]
    assert len(shapes_props) == len(shapes_sims)
    for got, want in zip(shapes_props, shapes_sims):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want))


def test_overlap_shapes_from_props_with_explicit_pairs_take_the_exact_path():
    """Given pairs (post-registration), the exact test runs - it needs sims, which are built
    from the same properties, per pair that reaches it, rather than for every source."""
    sources = [TiffImageSource(str(DATA_DIR / name)) for name in TIFF_FILES[:2]]
    translations = [{'x': 0.0, 'y': 0.0}, {'x': 5.0, 'y': 3.0}]

    sims = [build_source_shape_sim(source, 'yx', translation, None, 'source_metadata')
            for source, translation in zip(sources, translations)]
    props = [build_source_stack_props(source, 'yx', translation, None, 'source_metadata')
             for source, translation in zip(sources, translations)]

    shapes_sims, pairs_sims = create_overlap_shapes(sims, 'source_metadata', pairs=[(0, 1)])
    shapes_props, pairs_props = create_overlap_shapes(props, 'source_metadata', pairs=[(0, 1)])

    assert [tuple(pair) for pair in pairs_props] == [tuple(pair) for pair in pairs_sims]
    for got, want in zip(shapes_props, shapes_sims):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want))
