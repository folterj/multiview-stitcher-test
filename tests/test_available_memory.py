"""_available_memory() must read this job's own allocation, not the machine's hardware.

Same intent as constants._available_cpus reading the process cpuset: on a shared HPC node the
allocation is what a memory budget has to fit inside, and it is usually far smaller than what
the node physically has.
"""
import os

import pytest

from muvis_align.constants import _available_memory, default_fusion_chunk_bytes


@pytest.fixture
def no_slurm(monkeypatch):
    for variable in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU'):
        monkeypatch.delenv(variable, raising=False)


def test_reads_slurm_per_node_allocation(monkeypatch, no_slurm):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '65536')       # MB, as SLURM reports it
    assert _available_memory() == 64 * 1024 ** 3


def test_slurm_per_node_wins_over_per_cpu(monkeypatch, no_slurm):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '65536')
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '1024')
    assert _available_memory() == 64 * 1024 ** 3


def test_scales_slurm_per_cpu_by_allocated_cores(monkeypatch, no_slurm):
    from muvis_align import constants

    monkeypatch.setenv('SLURM_MEM_PER_CPU', '4096')
    assert _available_memory() == 4 * 1024 ** 3 * constants._available_cpus


def test_ignores_unparseable_slurm_value(monkeypatch, no_slurm):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', 'unlimited')
    # falls through to the real machine rather than raising
    assert _available_memory() is None or _available_memory() > 0


def test_falls_back_to_the_machine_without_slurm(no_slurm):
    memory = _available_memory()
    # every platform this package supports can answer this; a plausible floor guards a silently
    # wrong unit (bytes vs kB vs pages)
    assert memory is not None
    assert memory > 256 * 1024 ** 2


def test_derived_chunk_budget_is_plausible():
    assert 64 * 1024 ** 2 <= default_fusion_chunk_bytes <= 4 * 1024 ** 3
