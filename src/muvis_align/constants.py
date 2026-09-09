import os

import zarr

from muvis_align.zarr_compat import apply_windows_atomic_write_retry

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
def _available_memory():
    """Total memory this process may actually use, in bytes - the counterpart to
    _available_cpus above, and read in the same spirit: what was *allocated* to this job, not
    what the machine happens to have. A 2TB HPC node handed a 64GB job allocation must budget
    against the 64GB, so the batch-system and cgroup limits are checked before the hardware.
    Returns None if nothing here can tell, leaving callers to fall back to a fixed default.
    """
    # SLURM's own allocation, in MB (SLURM_MEM_PER_NODE wins; SLURM_MEM_PER_CPU is per
    # allocated core, so scale it by the cpuset _available_cpus already reads)
    for variable, multiplier in (('SLURM_MEM_PER_NODE', 1), ('SLURM_MEM_PER_CPU', _available_cpus)):
        value = os.environ.get(variable)
        if value:
            try:
                return int(float(value)) * multiplier * 1024 ** 2
            except ValueError:
                pass
    # container/cgroup limit (v2 then v1) - 'max', or an implausibly huge sentinel, means unset
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open(path) as file:
                limit = int(file.read().strip())
            if 0 < limit < 1 << 60:
                return limit
        except (OSError, ValueError):
            pass
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError):
        pass
    try:
        # Windows has no sysconf - ask the kernel directly rather than depend on psutil, which
        # is not one of this package's declared dependencies
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    return None


_available_memory_bytes = _available_memory()
# Per-output-chunk memory budget for fusion (see image.util.get_chunk_sizes). Fusion holds
# fusion_stack_arrays float32 arrays of one chunk's shape, times the sources overlapping that
# chunk - and dask's threaded scheduler runs one such chunk per worker at once, so the process
# peak is roughly this times the worker count. Budgeting a quarter of the allocation across
# those workers therefore leaves three quarters for everything else (source data, napari, the
# fused result itself), and scales the way the machine does: a 2TB/64-core node lands at the
# ceiling below, a 16GB laptop at a fraction of it. Sized per worker rather than as one global
# pool because the workers genuinely each hold a chunk simultaneously.
#
# The ceiling matters as much as the budget: past a few GB a chunk stops being a useful unit of
# parallel work (one task holding a whole zoomed-out view leaves 63 cores idle - the failure
# mode this budget exists to avoid), so extra headroom is spent on more chunks, not bigger ones.
default_fusion_chunk_bytes = min(4 * 1024 ** 3, max(
    64 * 1024 ** 2,
    int((_available_memory_bytes or 16 * 1024 ** 3) * 0.25 / max(1, _available_cpus))))
# multiview_stitcher's fusion holds this many same-shaped float32 arrays per output chunk at
# once: the stack of every overlapping source transformed into the chunk's grid, the matching
# blending-weight stack, and their product (fusion._core's field_ims_t / field_ws_t). Used by
# get_chunk_sizes() to size chunks against fusion's real peak memory rather than the output's
# own byte size, which for thousands of sources differ by orders of magnitude.
fusion_stack_arrays = 3
# get_contrast_limits() computes a real min/max off the coarsest pyramid level. That level is
# lazy, so the compute runs its whole fusion graph - fine when it is a handful of tasks, but
# above this many it is no longer the "cheap, up-front" step it is meant to be and a naive
# dtype-range guess is used instead (the user can auto-contrast from napari's own UI).
default_contrast_limits_max_tasks = 4096


# init_sources() constructs one ImageSource per file, each mostly waiting on a file
# open/header read rather than doing real CPU work - a thread pool overlaps that I/O latency
# (dominant on slow/network storage, e.g. a shared HPC filesystem) instead of paying it out
# serially file by file. Threads blocked on I/O don't consume CPU, so this is deliberately not
# capped AT core count (confirmed on a 4733-source, 32-worker run: wall time was ~32x less than
# the summed per-file time, i.e. near-perfectly I/O-bound, not GIL/CPU-bound) - but it still
# scales with it up to a fixed ceiling, so a genuinely small/constrained machine (few cores,
# likely also a modest network link) doesn't default to the same 64 threads a big one would.
default_source_init_workers = min(64, _available_cpus * 8)
# zarr v3 routes all of its own I/O through one shared, process-wide asyncio event loop plus a
# single internal ThreadPoolExecutor (zarr.core.sync._get_executor()), sized by this config value
# (default None -> Python's own min(32, cpu_count()+4)) - completely independent of
# default_source_init_workers above. Without raising it, ZarrImageSource reads stay bottlenecked
# on zarr's own smaller/default-sized pool no matter how many of our own worker threads are
# waiting to submit a read, which is why OME-Zarr sources parallelize noticeably worse than
# OME-TIFF ones (tifffile's own reads don't go through this at all). Set once, globally, here
# (not inside a `with` block) so it applies for the life of the process.
zarr.config.set({'threading.max_workers': default_source_init_workers})
# Windows only: zarr renames each metadata document into place, which fails outright if anything
# holds the destination open for the instant that takes. Applied here alongside the config above,
# for the same reason - it has to be in effect for the life of the process, before any store is
# written. See zarr_compat for what was measured.
apply_windows_atomic_write_retry()
# per-source preview/fusion prep (building each source's own fuse graph, gathering contrast
# limits/metadata) is genuine CPU-bound work, not I/O wait - unlike default_source_init_workers
# above there's no file-handle concern capping it, so this uses every allocated core
default_preview_workers = _available_cpus
# the interactive napari preview fuses this many sources' full native pyramids into one on-screen
# overview - matches MVSRegistration.create_preview()'s own default 'preview_scale' (16), so an
# on-screen preview stays proportionate to an exported one rather than fusing at native/scale0
# resolution regardless of how large or how many sources there are
default_interactive_preview_scale = 16
# ...and an upper bound on what that preview may cost regardless of how it was reached. The
# preview is a few hundred pixels on screen however large the fused stack behind it is, so
# fusing more than this to draw it is wasted: a run whose pre-processing scale was 1 fused
# 396.9GB and took 55 minutes to show what an 8x-reduced one showed in 9. preview_scale alone
# cannot prevent that - it selects a level relative to each source's own pyramid, and the
# post-pre-processing preview does not go through it at all - so the guard is on the resulting
# size instead (see image.util.reduce_msims_to_fused_size).
default_preview_max_bytes = 4 * 1024 ** 3

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
