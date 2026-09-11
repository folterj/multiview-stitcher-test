"""Which pyramid level a target scale selects - see get_level_from_scale()."""
from types import SimpleNamespace

from muvis_align.image.util import get_level_from_scale


def make_source(factors, reduce_z=False):
    """A source whose pyramid reduces x/y by `factors` - and z too, for a real z-stack."""
    return SimpleNamespace(
        scale_factors=[{'z': factor if reduce_z else 1.0, 'y': factor, 'x': factor}
                       for factor in factors],
        get_pixel_size=lambda: {'z': 1.0, 'y': 0.1, 'x': 0.1},
    )


def test_exact_target_selects_that_level():
    source = make_source([1, 2, 4, 8, 16])

    assert [get_level_from_scale(source, target)[0] for target in (1, 2, 4, 8, 16)] == [0, 1, 2, 3, 4]


def test_target_between_levels_selects_the_coarsest_level_still_fine_enough():
    """Never coarser than asked: a 6x request takes the 4x level, not the 8x or 16x one.

    A size-1 'z' keeps a factor of 1 at every level, so judging a level on *any* of its dims let
    even the coarsest pass on z alone - a 6x request loaded 16x data. Only the dims a pyramid
    actually reduces decide.
    """
    source = make_source([1, 2, 4, 8, 16])

    assert get_level_from_scale(source, 6)[0] == 2      # the 4x level
    assert get_level_from_scale(source, 12)[0] == 3     # the 8x level


def test_target_beyond_the_pyramid_selects_its_coarsest_level():
    source = make_source([1, 2, 4])

    level, residual, _ = get_level_from_scale(source, 32)

    assert level == 2
    # ...and the shortfall is reported, so callers can say the preview costs more than intended
    assert residual['x'] == 8


def test_a_z_stack_is_judged_on_z_as_well():
    """Here z does reduce, so it counts: a 3x request cannot take the 4x level, which would be
    coarser than asked in every dim including z."""
    source = make_source([1, 2, 4], reduce_z=True)

    assert get_level_from_scale(source, 3)[0] == 1
    assert get_level_from_scale(source, 4)[0] == 2


def test_single_level_source_always_selects_level_zero():
    source = make_source([1])

    assert get_level_from_scale(source, 16)[0] == 0
