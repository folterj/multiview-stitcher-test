"""ZarrImageSource must read metadata without building a msim - and get identical answers.

Project load only ever reads shapes, pixel sizes, origin, dtype and channels off a source, but
init_metadata used to build the whole msim (one xarray sim per pyramid level, plus one
zarr.json read per level) to obtain them: ~100ms per source, minutes across a few thousand.
The metadata now comes off the store's own consolidated metadata in a single read, with
ngff_zarr's version-aware parse as the fallback.

Both paths must agree exactly with what the msim would have reported - that equivalence is what
makes the fast path safe, so it is asserted here rather than argued from the NGFF spec.
"""
import numpy as np
import pytest
from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.ome_zarr_helper import save_ome_multiscale_levels
from muvis_align.image.ome_zarr_util import (_read_consolidated_ome_zarr_metadata,
                                             _read_ngff_ome_zarr_metadata,
                                             read_ome_zarr_source_metadata)
from muvis_align.image.source_helper import create_image_source

# (label, dim_order, shape) - the layouts a source can actually arrive in
LAYOUTS = [
    ('2d', 'yx', (256, 192)),
    ('2d forced t/c', 'tcyx', (1, 1, 256, 192)),
    ('multichannel', 'cyx', (3, 256, 192)),
    ('3d', 'zyx', (4, 256, 192)),
    ('3d multichannel', 'czyx', (2, 4, 256, 192)),
    ('full', 'tczyx', (1, 2, 4, 256, 192)),
]


def write_store(path, dim_order, shape, pixel_size=None, translation=None, levels=2):
    """A real multi-level OME-Zarr, written the way convert writes one."""
    spatial = [dim for dim in dim_order if dim in 'zyx']
    pixel_size = pixel_size or {dim: 0.5 for dim in spatial}
    translation = translation or {dim: 3.0 for dim in spatial}
    data = np.zeros(shape, dtype=np.uint16)
    written = []
    for level in range(levels):
        factor = 2 ** level
        slicing = tuple(slice(None, None, factor if dim in 'yx' else 1) for dim in dim_order)
        written.append((data[slicing],
                        {dim: pixel_size[dim] * (factor if dim in 'yx' else 1)
                         for dim in spatial}))
    save_ome_multiscale_levels(str(path), written, dim_order, [], translation)
    return str(path)


def metadata_via_msim(path):
    """What init_metadata used to derive, straight off the eagerly-built msim."""
    msim = ngff_utils.read_msim_from_ome_zarr(path, array_backend='dask',
                                              transform_key='affine_metadata')
    images = [msim[key].ds['image'] for key in msi_utils.get_sorted_scale_keys(msim)]
    image0 = images[0]
    return {'dimension_order': ''.join(image0.dims),
            'shapes': [tuple(image.shape) for image in images],
            'dtype': image0.dtype,
            'pixel_sizes': [si_utils.get_spacing_from_sim(image) for image in images],
            'position': si_utils.get_origin_from_sim(image0),
            'nchannels': image0.sizes.get('c', 1)}


def compare(fast, reference):
    assert fast['dimension_order'] == reference['dimension_order']
    assert [tuple(shape) for shape in fast['shapes']] == reference['shapes']
    assert fast['dtype'] == reference['dtype']
    assert fast['nchannels'] == reference['nchannels']
    assert len(fast['pixel_sizes']) == len(reference['pixel_sizes'])
    for got, want in zip(fast['pixel_sizes'], reference['pixel_sizes']):
        assert set(got) == set(want)
        for dim in want:
            assert got[dim] == pytest.approx(float(want[dim]))
    assert set(fast['position']) == set(reference['position'])
    for dim in reference['position']:
        assert fast['position'][dim] == pytest.approx(float(reference['position'][dim]))


@pytest.mark.parametrize('label, dim_order, shape', LAYOUTS)
def test_consolidated_fast_path_matches_the_msim(tmp_path, label, dim_order, shape):
    path = write_store(tmp_path / 'store.ome.zarr', dim_order, shape)
    fast = _read_consolidated_ome_zarr_metadata(path)
    assert fast is not None, 'consolidated fast path should apply to a freshly written v0.5 store'
    compare(fast, metadata_via_msim(path))


@pytest.mark.parametrize('label, dim_order, shape', LAYOUTS)
def test_ngff_fallback_path_matches_the_msim(tmp_path, label, dim_order, shape):
    path = write_store(tmp_path / 'store.ome.zarr', dim_order, shape)
    compare(_read_ngff_ome_zarr_metadata(path), metadata_via_msim(path))


def test_non_default_spacing_and_origin_survive(tmp_path):
    path = write_store(tmp_path / 'store.ome.zarr', 'zyx', (4, 256, 192),
                       pixel_size={'z': 7.0, 'y': 0.25, 'x': 0.125},
                       translation={'z': -2.0, 'y': 11.5, 'x': 4.25})
    reference = metadata_via_msim(path)
    compare(_read_consolidated_ome_zarr_metadata(path), reference)
    compare(_read_ngff_ome_zarr_metadata(path), reference)
    assert _read_consolidated_ome_zarr_metadata(path)['position'] == pytest.approx(
        {'z': -2.0, 'y': 11.5, 'x': 4.25})


def test_falls_back_when_consolidated_metadata_is_unusable(tmp_path, monkeypatch):
    path = write_store(tmp_path / 'store.ome.zarr', 'yx', (256, 192))
    reference = metadata_via_msim(path)

    # a store the fast path declines (here: no consolidated metadata at all) must still be read
    monkeypatch.setattr('muvis_align.image.ome_zarr_util._read_consolidated_ome_zarr_metadata',
                        lambda _path: None)
    compare(read_ome_zarr_source_metadata(path), reference)


def test_fast_path_declines_a_missing_or_v2_store(tmp_path):
    assert _read_consolidated_ome_zarr_metadata(str(tmp_path / 'nope.ome.zarr')) is None
    (tmp_path / 'v2.ome.zarr').mkdir()
    (tmp_path / 'v2.ome.zarr' / 'zarr.json').write_text('{"zarr_format": 2}')
    assert _read_consolidated_ome_zarr_metadata(str(tmp_path / 'v2.ome.zarr')) is None


@pytest.mark.parametrize('label, dim_order, shape', LAYOUTS)
def test_source_reports_the_same_metadata_and_builds_no_msim(tmp_path, label, dim_order, shape):
    path = write_store(tmp_path / 'store.ome.zarr', dim_order, shape)
    source = create_image_source(path)
    reference = metadata_via_msim(path)

    assert source.dimension_order == reference['dimension_order']
    assert [tuple(s) for s in source.shapes] == reference['shapes']
    assert source.dtype == reference['dtype']
    assert len(source.channels) == reference['nchannels']
    for got, want in zip(source.pixel_sizes, reference['pixel_sizes']):
        for dim in want:
            assert got[dim] == pytest.approx(float(want[dim]))

    # the whole point: no msim was constructed during init
    assert source._msim is None
    # and it still builds correctly, with this run's own transform re-stamped, on first access
    assert source.msim is not None
    assert source._msim is not None
    keys = msi_utils.get_sorted_scale_keys(source.msim)
    assert [tuple(source.msim[key].ds['image'].shape) for key in keys] == reference['shapes']


def test_level_data_still_reads_real_pixels(tmp_path):
    path = write_store(tmp_path / 'store.ome.zarr', 'yx', (256, 192))
    source = create_image_source(path)
    assert source._msim is None
    data = source.get_level_data(0)
    assert tuple(data.shape) == source.shapes[0]
    assert np.asarray(data).sum() == 0        # written as zeros


def test_real_v04_store_declines_the_fast_path_but_still_matches(tmp_path):
    """v0.4 stores have no consolidated zarr-v3 root, so they must take ngff_zarr's own parse -
    still without building the msim, and still with identical answers."""
    path = str(tmp_path / 'v04.ome.zarr')
    data = np.zeros((256, 192), dtype=np.uint16)
    save_ome_multiscale_levels(path, [(data, {'y': 0.5, 'x': 0.5}),
                                      (data[::2, ::2], {'y': 1.0, 'x': 1.0})],
                               'yx', [], {'y': 3.0, 'x': 3.0}, ome_version='0.4')

    assert _read_consolidated_ome_zarr_metadata(path) is None
    compare(read_ome_zarr_source_metadata(path), metadata_via_msim(path))

    source = create_image_source(path)
    assert source._msim is None
