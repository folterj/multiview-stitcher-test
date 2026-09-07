import os

zarr_extension = '.ome.zarr'
tiff_extension = '.ome.tiff'

default_ome_zarr_version = '0.5'

default_chunk_size = 1024
try:
    # sched_getaffinity (Linux-only) reads the process' real cpuset, which on a SLURM node
    # reflects the job's actual allocation - unlike os.cpu_count(), which reports the whole node
    # regardless of what was allocated to this job.
    _available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    _available_cpus = os.cpu_count() or 8
# init_sources() constructs one ImageSource per file, each mostly waiting on a file
# open/header read rather than doing real CPU work - a thread pool overlaps that I/O latency
# (dominant on slow/network storage, e.g. a shared HPC filesystem) instead of paying it out
# serially file by file. Threads blocked on I/O don't consume CPU, so this is deliberately not
# capped AT core count (confirmed on a 4733-source, 32-worker run: wall time was ~32x less than
# the summed per-file time, i.e. near-perfectly I/O-bound, not GIL/CPU-bound) - but it still
# scales with it up to a fixed ceiling, so a genuinely small/constrained machine (few cores,
# likely also a modest network link) doesn't default to the same 64 threads a big one would.
default_source_init_workers = min(64, _available_cpus * 8)
# per-source preview/fusion prep (building each source's own fuse graph, gathering contrast
# limits/metadata) is genuine CPU-bound work, not I/O wait - unlike default_source_init_workers
# above there's no file-handle concern capping it, so this uses every allocated core
default_preview_workers = _available_cpus
# the interactive napari preview fuses this many sources' full native pyramids into one on-screen
# overview - matches MVSRegistration.create_preview()'s own default 'preview_scale' (16), so an
# on-screen preview stays proportionate to an exported one rather than fusing at native/scale0
# resolution regardless of how large or how many sources there are
default_interactive_preview_scale = 16
# update_views()'s automatic fused-data preview (real blending, not the cheap 'compose' path) -
# above this many sources, skip building it automatically and show shapes only. multiview_stitcher's
# own fuse() takes a "per-chunk delayed" graph-construction path (its docs: "All-zarr inputs use
# map_blocks (thin graph); dask inputs use per-chunk delayed") once inputs have been through any of
# our own preprocessing/copying, as they always have here - one Python-level delayed task per
# (output chunk x overlapping source), which for thousands of sources across a large volume can
# reach millions of graph nodes well before any pixel is actually computed. Observed in practice:
# 4733 sources fusing into a 72x6800x6800 canvas drove memory to ~500GB and killed the process.
default_max_auto_preview_sources = 500

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
