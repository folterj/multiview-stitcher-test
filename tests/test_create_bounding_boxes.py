import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.util import (
    _minimal_bb_vertices,
    create_overlap_shapes,
    create_image_shapes,
    set_oriented_bounding_box_edges,
)


def test_create_overlap_shapes_projects_singleton_z_planes_to_2d():
    def make_sim(z, x):
        return si_utils.get_sim_from_array(
            np.zeros((1, 8, 8), dtype=np.uint8),
            dims=list('zyx'),
            scale={'z': 1, 'y': 1, 'x': 1},
            translation={'z': z, 'y': 0, 'x': x},
            transform_key='source_metadata',
        )

    z_levels = [-3.5, 7.25, 42]
    sims = [
        make_sim(z, x)
        for z in z_levels
        for x in (0, 4)
    ]

    shapes, pairs = create_overlap_shapes(
        sims,
        transform_key='source_metadata',
        force_2d=True,
    )

    assert np.asarray(pairs).tolist() == [[0, 1], [2, 3], [4, 5]]
    assert len(shapes) == len(z_levels)
    assert all(np.asarray(shape).shape[1] == 3 for shape in shapes)
    for shape, expected_z in zip(shapes, z_levels):
        np.testing.assert_allclose(np.asarray(shape)[:, 0], expected_z)

DATASETS = {
'0': np.array([[-1.51072116e+03, 2.53402693e+00, 2.53957371e+02],
     [-3.29915106e+02, 8.18186929e+01, 1.34088300e+04],
     [-1.09147618e+03, 1.32033584e+04, 1.36763537e+02],
     [ 8.93298765e+01, 1.32826430e+04, 1.32916362e+04],
     [-3.71880824e+02, -3.45399446e+01, 1.51956371e+02],
     [ 8.08925232e+02, 4.47447214e+01, 1.33068290e+04],
     [ 4.73641589e+01, 1.31662844e+04, 3.47625372e+01],
     [ 1.22817022e+03, 1.32455691e+04, 1.31896352e+04]]),

'1': np.array([[ -133.29726076, 11487.2102273,    -24.96843459],
     [   34.40461791, 11492.86579397, 13181.96565681],
     [ -121.72039954, 24695.20387892,   -30.77145818],
     [   45.98147913, 24700.85944559, 13176.16263322],
     [ 1010.61008185, 11486.20120614,   -39.49335782],
     [ 1178.31196052, 11491.85677282, 13167.44073358],
     [ 1022.18694307, 24694.19485776,   -45.29638141],
     [ 1189.88882175, 24699.85042444, 13161.63770999]]),

'2': np.array([[-4.59287824e+02,  6.83382917e+02,  1.21555299e+04],
     [ 7.34707344e+02,  6.55815383e+02,  2.53094220e+04],
     [-2.15067043e+01,  1.38841202e+04,  1.21434576e+04],
     [ 1.17248846e+03,  1.38565527e+04,  2.52973498e+04],
     [ 6.79397016e+02,  6.45525679e+02,  1.20520907e+04],
     [ 1.87339218e+03,  6.17958144e+02,  2.52059829e+04],
     [ 1.11717814e+03,  1.38462630e+04,  1.20400185e+04],
     [ 2.31117330e+03,  1.38186955e+04,  2.51939106e+04]]),

'3': np.array([[    0., 12000., 12000.],
     [    0., 12000., 25208.],
     [    0., 25208., 12000.],
     [    0., 25208., 25208.],
     [ 1144., 12000., 12000.],
     [ 1144., 12000., 25208.],
     [ 1144., 25208., 12000.],
     [ 1144., 25208., 25208.]]),

'[0 1]': np.array([[ 3.44046179e+01, 1.14928658e+04, 1.31819657e+04],
     [ 1.16910249e+03,  1.14918649e+04,  1.31675577e+04],
     [-1.30849940e+02,  1.31720860e+04,  5.07244027e+01],
     [ 4.73641589e+01,  1.31662844e+04,  3.47625372e+01],
     [-1.32205624e+02,  1.14872470e+04,  6.10006477e+01],
     [-5.96406097e+00,  1.14871314e+04,  4.96696650e+01],
     [ 2.88114348e+01,  1.32785796e+04,  1.26174252e+04],
     [ 3.47692601e+01,  1.19088846e+04,  1.31817829e+04],
     [ 7.93656249e+01,  1.32819740e+04,  1.31806286e+04],
     [ 1.17222316e+03,  1.32418125e+04,  1.25663522e+04],
     [ 1.17856740e+03,  1.17832825e+04,  1.31673127e+04],
     [ 1.17985026e+03,  1.32468998e+04,  1.31666696e+04]]),

'[0 2]': np.array([[   89.32987653, 13282.64303271, 13291.63620882],
     [ 1201.8817756,  13246.42485876, 13191.98975027],
     [  -13.90914387, 13275.71109736, 12141.49291388],
     [ 1097.05688221, 13239.53198656, 12040.57332155],
     [ -310.93003703,   679.60417522, 13403.52304293],
     [  793.09293777,   642.90061241, 13304.64508208],
     [ -422.91963253,   682.17380377, 12152.22614566],
     [  679.39701585,   645.52567852, 12052.09068516]]),

'[0 3]': np.array([[4.85944162e+01, 1.20000000e+04, 1.33030232e+04],
     [1.14400000e+03, 1.20000000e+04, 1.32046976e+04],
     [8.93298765e+01, 1.32826430e+04, 1.32916362e+04],
     [1.14400000e+03, 1.32483092e+04, 1.31971740e+04],
     [0.00000000e+00, 1.20000000e+04, 1.27604670e+04],
     [1.13686838e-13, 1.20000000e+04, 1.20000000e+04],
     [0.00000000e+00, 1.32766450e+04, 1.22964489e+04],
     [0.00000000e+00, 1.32740132e+04, 1.20000000e+04],
     [1.14400000e+03, 1.20000000e+04, 1.27025984e+04],
     [1.08107130e+03, 1.20000000e+04, 1.20000000e+04],
     [1.14400000e+03, 1.32399175e+04, 1.22519296e+04],
     [1.12138633e+03, 1.32383991e+04, 1.20000000e+04]]),

'[1 2]': np.array([[   23.27433547, 13882.63143245, 12139.38968869],
     [ 1117.1781359,  13846.26300461, 12040.01845906],
     [   34.40461791, 11492.86579397, 13181.96565681],
     [ 1141.36224818, 11491.88936553, 13167.90990771],
     [   21.10486203, 11492.41727392, 12134.57727503],
     [ 1039.08570933, 11491.48441055, 12042.17193117],
     [   30.61772627, 13882.91676954, 12717.69709908],
     [   35.51815461, 12763.29526136, 13181.40748481],
     [   72.62304483, 13881.94693055, 13180.45728935],
     [ 1174.02392033, 13844.95052181, 12666.27167939],
     [ 1179.314894,   12636.09947781, 13166.93800261],
     [ 1180.37467325, 13845.19728865, 13166.4067771 ]]),

'[1 3]': np.array([[1.98437784e+01, 1.20000000e+04, 1.20000000e+04],
     [1.14400000e+03, 1.20000000e+04, 1.20000000e+04],
     [3.48491231e+01, 1.20000000e+04, 1.31817428e+04],
     [1.14400000e+03, 1.20000000e+04, 1.31676588e+04],
     [3.10465456e+01, 2.47003558e+04, 1.20000000e+04],
     [1.14400000e+03, 2.46993803e+04, 1.20000000e+04],
     [4.59814791e+01, 2.47008594e+04, 1.31761626e+04],
     [1.14400000e+03, 2.46998909e+04, 1.31622204e+04]]),

'[2 3]': np.array([[    0.,         13883.40522169, 12141.50395078],
     [    0.,         12000.,         12137.55676635],
     [    0.,         13883.62368633, 12380.39065033],
     [ 1144.,         13857.21046431, 24983.50082325],
     [    0.,         12000.,         13069.769582  ],
     [ 1144.,         12000.,         13010.98583358],
     [ 1055.94980705, 12000.,         12041.70688781],
     [ 1144.,         13845.64372851, 12335.50701526],
     [ 1117.1781359,  13846.26300461, 12040.01845906],
     [ 1144.,         12000.,         25208.        ],
     [ 1144.,         13243.80041148, 25208.        ],
     [ 1102.64804062, 12000.,         25208.        ]])
}

DATASETS_2D = {
    label: points[:, 1:] for label, points in DATASETS.items()
}
DATASET_CASES = [
    pytest.param(points, id=label) for label, points in DATASETS.items()
]


@pytest.mark.parametrize("points", DATASET_CASES)
def test_minimal_bb_vertices_3d_always_has_eight_corners(points):
    shape = _minimal_bb_vertices(points)

    assert shape.shape == (8, 3)
    assert np.unique(np.round(shape, decimals=8), axis=0).shape[0] == 8


def test_create_image_shapes_matches_oriented_box_for_3d():
    points = DATASETS["0"]
    stack_props = {'shape': {'z': 1, 'y': 4, 'x': 4}, 'spacing': {'z': 1.0, 'y': 1.0, 'x': 1.0},
                   'origin': {'z': 0, 'y': 0.0, 'x': 0.0}}

    with patch("muvis_align.image.util.mv_graph.get_vertices_from_stack_props", return_value=points):
        shape = create_image_shapes([stack_props], transform_key=None, force_2d=False)[0]

    expected = _minimal_bb_vertices(points)
    np.testing.assert_allclose(shape, expected)


def test_minimal_bb_vertices_3d_preserves_rotated_orientation():
    points = DATASETS['0']

    shape = _minimal_bb_vertices(points)
    box_edges = shape[[1, 3, 4]] - shape[0]

    # A rotated input must not regress to the old world-axis-aligned box.
    assert any(np.count_nonzero(np.abs(edge) > 1e-6) > 1 for edge in box_edges)

    # Expressing every input point in the box basis verifies enclosure.
    coefficients = np.linalg.solve(box_edges.T, (points - shape[0]).T).T
    assert np.all(coefficients >= -1e-7)
    assert np.all(coefficients <= 1 + 1e-7)


@pytest.mark.parametrize("points", DATASET_CASES)
def test_napari_bounding_box_path_has_only_box_edges(points):
    shape = _minimal_bb_vertices(points)
    rendered_path = _minimal_bb_vertices(points, return_edge_path=True)
    basis_edges = shape[[1, 3, 4]] - shape[0]
    rendered_edges = np.diff(rendered_path, axis=0)

    assert all(
        any(
            np.allclose(edge, basis_edge)
            or np.allclose(edge, -basis_edge)
            for basis_edge in basis_edges
        )
        for edge in rendered_edges
    )


def test_oriented_edge_path_is_applied_to_napari_layer():
    shape = _minimal_bb_vertices(DATASETS["0"])
    shape_with_slice_coordinate = np.column_stack((np.zeros(8), shape))
    bounding_box = MagicMock()
    bounding_box.ndisplay = 3
    layer = MagicMock()
    layer._data_view.bounding_boxes = [bounding_box]

    set_oriented_bounding_box_edges(layer, [shape_with_slice_coordinate])

    np.testing.assert_allclose(
        bounding_box._set_meshes.call_args.args[0],
        _minimal_bb_vertices(shape[:, -3:], return_edge_path=True),
    )
    bounding_box._set_meshes.assert_called_once()
    layer._data_view._update_mesh_vertices.assert_called_once_with(0, edge=True)


def test_create_image_shapes_matches_oriented_box_for_force_2d():
    points = DATASETS['0']
    # create_image_shapes takes stack properties directly, so the geometry can be handed over
    # as-is rather than mocked onto a sim
    stack_props = {'shape': {'z': 1, 'y': 4, 'x': 4}, 'spacing': {'z': 1.0, 'y': 1.0, 'x': 1.0},
                   'origin': {'z': 0, 'y': 0.0, 'x': 0.0}}

    with patch("muvis_align.image.util.mv_graph.get_vertices_from_stack_props", return_value=points):
        shape = create_image_shapes([stack_props], transform_key=None, force_2d=True)[0]

    expected = _minimal_bb_vertices(points[:, 1:])
    np.testing.assert_allclose(shape, expected)
    assert shape.shape == (4, 2)


@pytest.mark.parametrize("points", DATASET_CASES)
def test_minimal_bb_vertices_2d_is_a_simple_non_crossing_rectangle(points):
    """Regression test: force_2d shapes must be a proper (non-self-intersecting)
    rectangle, not a bowtie caused by wrong corner ordering.
    """
    shape = _minimal_bb_vertices(points[:, 1:])

    assert shape.shape == (4, 2)


@pytest.mark.parametrize("points", DATASET_CASES)
def test_create_image_shapes_force_2d_is_simple_non_crossing_rectangle(points):
    stack_props = {'shape': {'z': 1, 'y': 4, 'x': 4}, 'spacing': {'z': 1.0, 'y': 1.0, 'x': 1.0},
                   'origin': {'z': 0, 'y': 0.0, 'x': 0.0}}

    with patch("muvis_align.image.util.mv_graph.get_vertices_from_stack_props", return_value=points):
        shape = create_image_shapes([stack_props], transform_key=None, force_2d=True)[0]

    shape = np.asarray(shape)
    assert shape.shape == (4, 2)
