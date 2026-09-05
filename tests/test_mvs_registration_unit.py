import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from muvis_align.MVSRegistration import MVSRegistration, RegState


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (RegState.UNINIT, (False, False, False, False)),
        (RegState.INIT, (True, False, False, False)),
        (RegState.PAIRS_REG, (True, True, False, False)),
        (RegState.GLOBAL_REG, (True, True, True, False)),
        (RegState.FUSED, (True, True, True, True)),
    ],
    ids=lambda value: value.name if isinstance(value, RegState) else None,
)
def test_registration_state_predicates(state, expected):
    registration = MVSRegistration()
    registration.state = state

    actual = (
        registration.is_initialised(),
        registration.is_pairs_registered(),
        registration.is_global_registered(),
        registration.is_fused(),
    )

    assert actual == expected


def test_reset_clears_registration_state():
    registration = MVSRegistration()
    registration.state = RegState.FUSED
    registration.msims = [object()]
    registration.metrics = {"quality": 1}

    registration.reset()

    assert registration.state is RegState.UNINIT
    assert registration.msims == []
    assert registration.register_msims is None
    assert registration.sources == []
    assert registration.metrics == {}
    assert registration.register_indices is None


def test_init_with_explicit_files_sets_labels_and_output(tmp_path):
    inputs = [str(tmp_path / "tile_01.tif"), str(tmp_path / "tile_02.tif")]

    registration = MVSRegistration()
    result = registration.init(
        operation="register",
        label="sample",
        input_path=inputs,
        input_labels=["left", "right"],
        output_path="results/",
    )

    assert result is True
    assert registration.state is RegState.INIT
    assert registration.filenames == [Path(path).as_posix() for path in inputs]
    assert registration.file_labels == ["left", "right"]
    assert registration.output == str(tmp_path / "results") + "/"


def test_init_returns_false_when_pattern_has_no_files(tmp_path):
    registration = MVSRegistration()

    with patch("muvis_align.MVSRegistration.dir_regex", return_value=[]):
        result = registration.init(
            input_path=str(tmp_path / "*.tif"),
            output_path="results/",
        )

    assert result is False


def test_init_params_normalises_sections_and_forwards_options():
    registration = MVSRegistration()
    params = {
        "operation": "register",
        "input": "images/*.tif",
        "output": "results/",
        "preprocessing": {"scale": 2},
        "registration": {"method": "phase"},
        "fusion": {"method": "max"},
    }
    general = {
        "overwrite": True,
        "clear": True,
        "ui": "napari",
        "verbose": True,
        "debug": True,
    }

    with patch.object(registration, "init", return_value=True) as init:
        result = registration.init_params(general, params, label="sample")

    assert result is True
    assert registration.input_params == {"path": "images/*.tif"}
    assert registration.output_params == {"path": "results/"}
    assert registration.preprocess_params == {"scale": 2}
    assert init.call_args.kwargs["overwrite"] is True
    assert init.call_args.kwargs["label"] == "sample"


def _make_msims(n=1, size=8, pixel_size=1.0):
    # small, real (not mocked) single-level msims - cheap enough that preprocess()'s actual
    # per-level operations (gaussian, normalisation) can just run for real instead of being mocked
    from multiview_stitcher import msi_utils, spatial_image_utils as si_utils
    msims = []
    for i in range(n):
        data = np.full((size, size), i + 1, dtype=np.uint16)
        sim = si_utils.get_sim_from_array(
            data, dims=['y', 'x'], scale={'x': pixel_size, 'y': pixel_size},
            translation={'x': 0, 'y': 0}, transform_key='source_metadata')
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
    return msims


@pytest.mark.parametrize(
    ("kwargs", "expected_modified"),
    [
        ({}, False),
        ({"scale": 1}, False),
        ({"normalisation": "none"}, False),
        ({"normalisation": False}, False),
        ({"gaussian_sigma": 1}, True),
    ],
)
def test_preprocess_sets_modified_flag_for_enabled_steps(kwargs, expected_modified):
    registration = MVSRegistration()
    registration.scales = [{"x": 1.0, "y": 1.0}]
    registration.source_transform_key = "source_metadata"
    msims = _make_msims()

    _, _, modified = registration.preprocess(msims, **kwargs)

    assert modified is expected_modified


def test_preprocess_applies_scale_via_select_msim_subpyramid():
    # preprocess()'s `scale` override selects a real (smaller) sub-pyramid directly from the
    # msims it's given (every native level at or coarser than `scale`), rather than resizing to
    # an exact match or re-running the whole init_data() pipeline a second time
    registration = MVSRegistration()
    registration.scales = [{"x": 1.0, "y": 1.0}]
    registration.source_transform_key = "source_metadata"
    registration.sources = [object()]
    msims = _make_msims()

    with patch(
        "muvis_align.MVSRegistration.select_msim_subpyramid_at_scale", return_value=msims,
    ) as select_msim_subpyramid_at_scale:
        _, _, modified = registration.preprocess(msims, scale=2)

    assert modified is True
    select_msim_subpyramid_at_scale.assert_called_once_with(msims, registration.sources, 2)


def test_validate_overlap_reports_near_images():
    registration = MVSRegistration()
    sims = [object(), object()]
    positions = [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.0}]

    with (
        patch(
            "muvis_align.MVSRegistration.get_sim_position_final",
            side_effect=positions,
        ),
        patch(
            "muvis_align.MVSRegistration.get_sim_physical_size",
            return_value={"x": 1.0, "y": 1.0},
        ),
    ):
        distances, overlaps = registration.validate_overlap(
            sims, ["left", "right"]
        )

    assert len(distances) == 2
    assert overlaps == [True, True]


def test_get_metrics_supports_summary_tuple_and_numpy_pair():
    registration = MVSRegistration()
    registration.metrics = {
        "summary": {"source": {"quality": 0.5}},
        "pairs": {(0, 1): {"registered": {"quality": 0.8}}},
    }

    assert registration.get_metrics("quality") == 0.5
    assert registration.get_metrics(
        "quality", np.array([0, 1])
    ) == pytest.approx(0.8)
    assert registration.get_metrics(pair=(3, 4)) == {}


def test_output_exists_supports_zarr_and_regular_files(tmp_path):
    registration = MVSRegistration()
    registration.output = str(tmp_path) + "/"
    zarr_output = tmp_path / "fused.zarr"
    zarr_output.mkdir()
    (zarr_output / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preview.tif").write_bytes(b"image")

    assert registration.output_exists("fused", ".zarr")
    assert registration.output_exists("preview", "tif")
    assert not registration.output_exists("missing", ".zarr")


@pytest.mark.parametrize(
    ("output_exists", "existing_path", "expected_state"),
    [
        (True, None, RegState.FUSED),
        (False, "mappings.json", RegState.GLOBAL_REG),
        (False, "pairs.json", RegState.PAIRS_REG),
    ],
)
def test_check_progress_uses_most_advanced_available_state(
    output_exists, existing_path, expected_state
):
    registration = MVSRegistration()
    registration.output = "output/"
    registration.output_params = {
        "pair_mappings": "pairs.json",
        "mappings": "mappings.json",
    }

    def path_exists(path):
        return existing_path is not None and path.endswith(existing_path)

    with (
        patch.object(
            registration, "output_exists", return_value=output_exists
        ),
        patch(
            "muvis_align.MVSRegistration.os.path.exists",
            side_effect=path_exists,
        ),
    ):
        registration.check_progress("fused", ".zarr")

    assert registration.state is expected_state


def test_init_data_defers_msim_construction_to_first_msims_read():
    """init_data() should only resolve cheap per-source metadata (position/scale/rotation) -
    the expensive per-source msim build (build_source_msim(), the actual bottleneck when
    loading many tiles) must not run until something genuinely reads reg.msims."""
    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[
            'data/S000/000_000_0.tiff',
            'data/S000/000_001_0.tiff',
        ],
        output_path='../../output/test_init_data_defers_msims/',
    )
    reg.init_data()

    # cheap metadata is already fully resolved...
    assert reg._msims is None  # ... but the expensive msims list is not built yet
    assert len(reg.positions) == 2
    assert len(reg.scales) == 2
    assert len(reg.rotations) == 2
    assert all(source._msim is None for source in reg.sources)  # per-source msim deferred too

    # reading reg.msims (the property) triggers the deferred build, on demand
    msims = reg.msims
    assert len(msims) == 2
    assert reg._msims is msims
    assert all(source._msim is not None for source in reg.sources)


def test_select_pair_overlap_then_register_overlap_matches_register_pairs():
    """select_pair_overlap()/register_overlap() together replace the old single register_pair()
    call, split so a caller (e.g. a UI preview) can cache the overlap crop select_pair_overlap()
    returns and re-run register_overlap() on it for parameter-only changes, without re-selecting
    resolution or re-cropping from the (possibly large) source data every time."""
    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[
            'data/S000/S000_000_000.ome.zarr',
            'data/S000/S000_000_001.ome.zarr',
        ],
        output_path='../../output/test_select_pair_overlap/',
    )
    reg.init_data()
    reg.preprocess(reg.msims)

    params = {'method': 'orb', 'pairing': 'orthogonal'}
    msim1, msim2 = reg.register_msims[0], reg.register_msims[1]

    overlap1, overlap2, sims_pixel_space = reg.select_pair_overlap(msim1, msim2, params=params)
    assert overlap1.ndim == 2
    assert overlap2.ndim == 2

    transform, quality, result = reg.register_overlap(overlap1, overlap2, sims_pixel_space, params=params)

    assert transform.shape[-2:] == (3, 3)  # 2D homogeneous affine
    assert not np.isnan(quality)

    # the raw pairwise_reg_func result must survive - feature-based methods report points/matches
    # used for a napari preview overlay
    assert 'affine_matrix' in result
    assert 'quality' in result
    assert 'fixed_points' in result
    assert 'moving_points' in result


def test_register_overlap_reuses_cached_overlap_across_param_changes():
    """The whole point of splitting select_pair_overlap()/register_overlap(): the same overlap
    crop can be registered again with different registration parameters, without recomputing the
    crop - the two calls below reuse the exact same overlap1/overlap2/sims_pixel_space."""
    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[
            'data/S000/S000_000_000.ome.zarr',
            'data/S000/S000_000_001.ome.zarr',
        ],
        output_path='../../output/test_register_overlap_cache/',
    )
    reg.init_data()
    reg.preprocess(reg.msims)

    msim1, msim2 = reg.register_msims[0], reg.register_msims[1]
    overlap1, overlap2, sims_pixel_space = reg.select_pair_overlap(
        msim1, msim2, params={'method': 'orb'}
    )

    transform1, quality1, result1 = reg.register_overlap(
        overlap1, overlap2, sims_pixel_space, params={'method': 'orb'}
    )
    transform2, quality2, result2 = reg.register_overlap(
        overlap1, overlap2, sims_pixel_space, params={'method': 'sift'}
    )

    # different methods, same crop - both must produce a valid result from the identical input
    assert transform1.shape == transform2.shape
    assert not np.isnan(quality1)
    assert not np.isnan(quality2)
