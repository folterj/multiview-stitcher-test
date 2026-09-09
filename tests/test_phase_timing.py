"""The threaded-phase timing line has to say which regime a run is in.

Reporting wall against the summed per-item time - the obvious summary, and what these logs used
to print - cannot distinguish "the threads overlapped real waiting" from "the threads queued
behind each other". Under a pool, an item's wall time also counts its wait for the GIL, so that
sum inflates with the worker count either way: measured on 328 sources, it went 15s -> 1028s
from 1 to 64 workers while wall time went 14.9s -> 16.6s, so the 60x "overlap" it implied was
worth nothing at all.
"""
import pytest

from muvis_align.util import format_phase_timing

CPU_BOUND = 'more workers will not help'
IO_BOUND = 'more workers can overlap it'


def test_the_real_measurement_that_motivated_this_reads_as_cpu_bound():
    # 328 sources at 64 workers: wall 16.6s, summed per-source 1027.9s, but only 16.5s of CPU.
    # Sequentially the same phase took 14.9s - i.e. the threads bought nothing.
    line = format_phase_timing(16.64, [1027.9 / 328] * 328, [16.5 / 328] * 328, 64)
    assert CPU_BOUND in line
    assert IO_BOUND not in line


def test_under_parallelised_io_says_more_workers_would_help():
    # each item waits 3.0s but needs only 0.05s of CPU; 8 workers cannot overlap it all
    line = format_phase_timing(328 * 3.0 / 8, [3.0] * 328, [0.05] * 328, 8)
    assert IO_BOUND in line
    assert '87%' in line


def test_the_same_io_work_with_enough_workers_reaches_the_floor():
    # total CPU is the floor: no arrangement of threads beats it, so this must not keep
    # advising more workers
    line = format_phase_timing(16.4, [3.0] * 328, [0.05] * 328, 64)
    assert CPU_BOUND in line


def test_reports_wall_total_and_cpu():
    line = format_phase_timing(10.0, [1.0] * 4, [0.5] * 4, 4)
    assert 'wall 10.0s' in line
    assert 'per-item total 4.0s' in line
    assert 'cpu 2.0s' in line
    assert 'with 4 workers' in line


def test_mean_and_max_come_from_wall_times():
    line = format_phase_timing(5.0, [1.0, 3.0], [0.1, 0.2], 2)
    assert 'mean 2000ms' in line
    assert 'max 3000ms' in line


def test_no_cpu_times_still_reports_without_a_verdict():
    # a caller that cannot measure CPU time must not get a made-up regime
    line = format_phase_timing(5.0, [1.0, 3.0], [], 2)
    assert 'wall 5.0s' in line
    assert CPU_BOUND not in line
    assert IO_BOUND not in line


@pytest.mark.parametrize('workers', [1, 8, 64])
def test_worker_count_is_reported(workers):
    assert f'with {workers} workers' in format_phase_timing(1.0, [1.0], [1.0], workers)
