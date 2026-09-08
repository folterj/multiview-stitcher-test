"""convert_to_um must cover every length unit OME can express, in both spellings.

Readers pass whichever form they hold: an OME Plane's PositionXUnit is an abbreviation ('nm'),
while ngff_zarr's _normalize_unit hands over the spelled-out NGFF name ('nanometer') for the
same file. A unit missing from the table is not an error - it silently scales by 1, i.e. mis-
places or mis-sizes the image with nothing to show for it. 'nanometer' was exactly that case.
"""
import logging

import pytest

from muvis_align.util import convert_to_um


# (unit, um per unit) - abbreviation and spelled-out name must agree
@pytest.mark.parametrize('unit, factor', [
    ('Å', 1e-4), ('A', 1e-4), ('angstrom', 1e-4),
    ('pm', 1e-6), ('picometer', 1e-6),
    ('nm', 1e-3), ('nanometer', 1e-3),
    ('µm', 1.0), ('um', 1.0), ('micrometer', 1.0), ('micron', 1.0),
    ('mm', 1e3), ('millimeter', 1e3),
    ('cm', 1e4), ('centimeter', 1e4),
    ('m', 1e6), ('meter', 1e6),
])
def test_known_units(unit, factor):
    assert convert_to_um(1.0, unit) == pytest.approx(factor)
    assert convert_to_um(2.5, unit) == pytest.approx(2.5 * factor)


@pytest.mark.parametrize('abbreviation, name', [
    ('nm', 'nanometer'), ('pm', 'picometer'), ('µm', 'micrometer'),
    ('mm', 'millimeter'), ('cm', 'centimeter'), ('m', 'meter'), ('Å', 'angstrom'),
])
def test_both_spellings_agree(abbreviation, name):
    assert convert_to_um(7.0, abbreviation) == pytest.approx(convert_to_um(7.0, name))


def test_every_unit_ngff_zarr_can_produce_is_covered():
    """ngff_zarr normalizes OME units to NGFF names before a source converts them, so every
    value it can emit has to resolve to a real factor."""
    from ngff_zarr.tiff_to_ngff_image import OME_UNIT_TO_NGFF

    from muvis_align.util import um_conversions

    for ome_unit, ngff_name in OME_UNIT_TO_NGFF.items():
        assert ngff_name in um_conversions, f'{ngff_name!r} (from OME {ome_unit!r}) is unhandled'
        assert ome_unit in um_conversions, f'OME unit {ome_unit!r} is unhandled'
        assert convert_to_um(1.0, ome_unit) == pytest.approx(convert_to_um(1.0, ngff_name))


def test_case_is_ignored():
    assert convert_to_um(1.0, 'Micrometer') == pytest.approx(1.0)
    assert convert_to_um(1.0, 'NM') == pytest.approx(1e-3)


def test_missing_unit_is_left_alone_and_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        assert convert_to_um(3.0, None) == 3.0
        assert convert_to_um(3.0, '') == 3.0
    assert 'Unrecognised' not in caplog.text


def test_unrecognised_unit_is_left_unscaled_but_logged(caplog):
    with caplog.at_level(logging.WARNING):
        assert convert_to_um(3.0, 'furlong') == 3.0
    assert 'furlong' in caplog.text
