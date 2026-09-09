"""Headless benchmark mirroring the napari Interface flow, on a local dataset.

Times the stages a napari project load / pre-processing / fusion preview goes through, without a
GUI, so the same measurements can be taken on a laptop-sized dataset instead of a full HPC run.
Point DATASETS at your own data; the defaults are the 328-source tiff and OME-Zarr copies of one
slide. Run as e.g. `python testing/local_perf_bench.py zarr --limit 40`.

Stages timed (same names as the HPC log lines):
  init sources            -> MVSRegistration.init_data (metadata only)
  build msims             -> ensure_msims  (the 'load image data' phase)
  preprocess              -> preprocess(scale=...)
  copy_transforms         -> copy_transforms_to_msims
  fuse (build graph)      -> MVSRegistration.fuse
  compute coarsest level  -> what napari actually pulls to draw the overview
"""
import argparse, logging, os, sys, time
import numpy as np

from multiview_stitcher import msi_utils
from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.image.util import copy_transforms_to_msims, get_msim_image0

DATASETS = {
    'tiff': dict(
        input_path='C:/Project/slides/12193/data/*/*.tiff',
        source_metadata={'position': {'z': 'fn[-4]', 'y': 'fn[-3]*24', 'x': 'fn[-2]*24'},
                         'scale': {'z': '1', 'y': '0.004', 'x': '0.004'},
                         'rotation': 'source'},
    ),
    'zarr': dict(
        input_path='C:/Project/slides/12193/data_zarr/*/*.zarr',
        source_metadata={'position': {'z': 'source', 'y': 'source', 'x': 'source'},
                         'scale': {'z': 'source', 'y': 'source', 'x': 'source'},
                         'rotation': 'source'},
    ),
}


class Stage:
    times = {}

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        Stage.times[self.name] = dt
        print(f'  {self.name:38s} {dt:8.2f} s', flush=True)


def run(kind, limit=None, scale=8, do_fuse=True):
    cfg = DATASETS[kind]
    print(f'=== {kind}', flush=True)
    reg = MVSRegistration()
    reg.verbose = True
    reg.logging_time = True
    ok = reg.init(operation='register', input_path=cfg['input_path'],
                  output_path=os.path.abspath(f'output_bench_{kind}') + '/', overwrite=True, verbose=True)
    assert ok
    if limit:
        reg.filenames = reg.filenames[:limit]
        reg.file_labels = reg.file_labels[:limit]
    print(f'  {len(reg.filenames)} files', flush=True)

    with Stage('init sources (metadata)'):
        reg.init_data(source_metadata=cfg['source_metadata'])

    with Stage('build msims (load image data)'):
        msims = reg.ensure_msims()

    with Stage('preprocess'):
        reg.preprocess(msims, scale=scale, normalisation=None, filter_foreground=False)

    pmsims = reg.register_msims
    lvl0 = get_msim_image0(pmsims[0])
    print(f'  preprocessed level0 shape {tuple(lvl0.shape)}'
          f' levels={len(msi_utils.get_sorted_scale_keys(pmsims[0]))}', flush=True)

    if not do_fuse:
        return

    with Stage('copy register_msims (deep)'):
        cmsims = [msim.copy(deep=True) for msim in pmsims]
    with Stage('copy_transforms_to_msims'):
        copy_transforms_to_msims(reg.msims, cmsims, reg.source_transform_key)

    with Stage('fuse (build graph)'):
        fused, _ = reg.fuse(cmsims, fusion_method='additive', transform_key=reg.source_transform_key)

    keys = msi_utils.get_sorted_scale_keys(fused)
    coarsest = fused[keys[-1]].ds['image'].data
    print(f'  fused levels={len(keys)} coarsest={coarsest.shape}', flush=True)
    with Stage('compute coarsest fused level'):
        np.asarray(coarsest)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=list(DATASETS))
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--scale', type=float, default=8)
    p.add_argument('--no-fuse', action='store_true')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s: %(message)s')
    t0 = time.time()
    run(a.kind, limit=a.limit, scale=a.scale, do_fuse=not a.no_fuse)
    print(f'  {"TOTAL":38s} {time.time()-t0:8.2f} s')
