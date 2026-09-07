import os

zarr_extension = '.ome.zarr'
tiff_extension = '.ome.tiff'

default_ome_zarr_version = '0.5'

default_chunk_size = 1024
# init_sources() constructs one ImageSource per file, each mostly waiting on a file
# open/header read rather than doing real CPU work - a thread pool overlaps that I/O latency
# (dominant on slow/network storage, e.g. a shared HPC filesystem) instead of paying it out
# serially file by file. Threads blocked on I/O don't consume CPU, so this is deliberately not
# capped by core count (confirmed on a 4733-source, 32-worker run: wall time was ~32x less than
# the summed per-file time, i.e. near-perfectly I/O-bound, not GIL/CPU-bound) - only by a fixed
# ceiling, to avoid opening an unreasonable number of file handles/connections against the
# filesystem at once.
try:
    # sched_getaffinity (Linux-only) reads the process' real cpuset, which on a SLURM node
    # reflects the job's actual allocation - unlike os.cpu_count(), which reports the whole node
    # regardless of what was allocated to this job. Only used by default_preview_workers below
    # (genuinely CPU-bound work); init_sources' own worker count no longer depends on this.
    _available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    _available_cpus = os.cpu_count() or 8
default_source_init_workers = 64
# per-source preview/fusion prep (building each source's own fuse graph, gathering contrast
# limits/metadata) is genuine CPU-bound work, not I/O wait - unlike default_source_init_workers
# above there's no file-handle concern capping it, so this uses every allocated core
default_preview_workers = _available_cpus
# the interactive napari preview fuses this many sources' full native pyramids into one on-screen
# overview - matches MVSRegistration.create_preview()'s own default 'preview_scale' (16), so an
# on-screen preview stays proportionate to an exported one rather than fusing at native/scale0
# resolution regardless of how large or how many sources there are
default_interactive_preview_scale = 16

prereg_mappings_name = 'prereg_mappings.csv'
default_pair_mappings_name = 'pair_mappings.json'
default_mappings_name = 'mappings.json'
default_mappings_tabular_name = 'mappings.csv'
original_positions_name = 'positions_original.pdf'
registered_positions_name = 'positions_registered.pdf'
metrics_name = 'metrics.json'

default_transform_key = 'transform'
default_quality_key = 'quality'

NAPARI_PROJECT_TEMPLATE = 'ui/project_template.yaml'
