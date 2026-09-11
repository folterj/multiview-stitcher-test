"""Resuming a saved pair registration builds its own graph - see build_pairs_graph()."""
import networkx as nx
import numpy as np
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.util import build_pairs_graph, make_msims_2d, msim_is_2d, wrap_sims_as_msims

TRANSFORM_KEY = 'affine_metadata'


def make_msim(translation, size=8, spatial_dims=('y', 'x'), z_size=1):
    """A source shaped like the real ones: t and c dims, then the spatial dims."""
    shape = [1, 1] + [z_size if dim == 'z' else size for dim in spatial_dims]
    array = np.zeros(shape, dtype=np.uint16)
    sim = si_utils.get_sim_from_array(
        array,
        dims=['t', 'c'] + list(spatial_dims),
        scale={dim: 1.0 for dim in spatial_dims},
        translation={dim: float(value) for dim, value in zip(spatial_dims, translation)},
        transform_key=TRANSFORM_KEY,
    )
    return wrap_sims_as_msims([sim])[0]


def flatten(stack_props):
    """stack_props as a flat list of numbers, whatever shape each entry happens to take."""
    values = []
    for key in sorted(stack_props):
        entry = stack_props[key]
        items = [entry[dim] for dim in sorted(entry)] if isinstance(entry, dict) else [entry]
        values.extend(float(np.asarray(item).ravel()[0]) for item in items)
    return values


def test_pairs_graph_has_the_nodes_edges_and_stack_props_the_graph_build_would_give():
    """What multiview_stitcher's build_view_adjacency_graph_from_msims() produces for a known
    set of pairs - a node per source carrying its stack properties, an edge per pair - without
    the linear program per pair it runs to rediscover overlaps we already have. (Checked against
    the real thing on a 328-source project: same nodes, same edges, same stack_props, 7x faster.)
    """
    msims = [make_msim((0, 0)), make_msim((0, 6)), make_msim((6, 0))]
    pairs = [(0, 1), (0, 2)]

    graph = build_pairs_graph(msims, pairs, TRANSFORM_KEY)

    assert set(graph.nodes) == {0, 1, 2}
    assert set(map(frozenset, graph.edges)) == {frozenset((0, 1)), frozenset((0, 2))}
    for node, msim in enumerate(msims):
        expected = si_utils.get_stack_properties_from_sim(
            msi_utils.get_sim_from_msim(msim), transform_key=TRANSFORM_KEY)
        props = graph.nodes[node]['stack_props']
        assert sorted(props) == sorted(expected)
        assert np.allclose(flatten(props), flatten(expected))


def test_pairs_graph_carries_an_overlap_weight_when_one_is_known():
    """Downstream reads the weight as .get('overlap', 1.0), so it is optional - but the saved
    bboxes give it for nothing, and passing it keeps the resolution methods that use it honest."""
    msims = [make_msim((0, 0)), make_msim((0, 6))]

    graph = build_pairs_graph(msims, [(0, 1)], TRANSFORM_KEY, overlaps={(0, 1): 16.0})
    without = build_pairs_graph(msims, [(0, 1)], TRANSFORM_KEY)

    assert graph.edges[0, 1]['overlap'] == 16.0
    assert without.edges[0, 1].get('overlap', 1.0) == 1.0


def test_pairs_graph_has_a_node_per_source_even_with_no_pairs():
    msims = [make_msim((0, 0)), make_msim((0, 6))]

    graph = build_pairs_graph(msims, [], TRANSFORM_KEY)

    assert isinstance(graph, nx.Graph)
    assert set(graph.nodes) == {0, 1}
    assert not graph.edges


def test_make_msims_2d_leaves_an_already_2d_msim_alone():
    """Rebuilding a msim's DataTree is minutes of xarray construction for a few thousand
    sources, so what is already 2D is passed through untouched."""
    msims = [make_msim((0, 0)), make_msim((0, 6))]
    assert all(msim_is_2d(msim) for msim in msims)

    converted = make_msims_2d(msims)

    assert [id(msim) for msim in converted] == [id(msim) for msim in msims]


def test_make_msims_2d_still_converts_a_3d_msim():
    msims = [make_msim((0, 0, 0), spatial_dims=('z', 'y', 'x'))]
    assert not msim_is_2d(msims[0])

    converted = make_msims_2d(msims)

    assert converted[0] is not msims[0]
    assert msim_is_2d(converted[0])
