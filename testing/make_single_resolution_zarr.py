"""Write a grid of single-resolution (one level, no pyramid) OME-Zarr tiles.

Source data of this shape is what leaves the pipeline nothing to reduce resolution with, so a
coarse preview ends up fusing at the store's native resolution. Use it with local_perf_bench.py
to measure that case (ZarrImageSource synthesizes the missing coarse levels - see its _load_data).

    python testing/make_single_resolution_zarr.py 4 6400
"""
import os, shutil, sys
import numpy as np
import zarr

OUT = os.path.abspath('flat_zarr')
GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
PIXEL = 0.004
OVERLAP = 0.9   # tile step as a fraction of tile extent

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
rng = np.random.default_rng(0)
base = rng.integers(0, 255, size=(SIZE, SIZE), dtype=np.uint8)

for row in range(GRID):
    for col in range(GRID):
        path = f'{OUT}/S000/tile_{row:03d}_{col:03d}.ome.zarr'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        root = zarr.open_group(path, mode='w', zarr_format=3)
        arr = root.create_array('scale0/image', shape=(1, 1, 1, SIZE, SIZE), chunks=(1, 1, 1, 512, 512),
                                dtype='uint8')
        arr[0, 0, 0] = np.roll(base, (row * 37, col * 53), axis=(0, 1))
        ty = row * SIZE * PIXEL * OVERLAP
        tx = col * SIZE * PIXEL * OVERLAP
        root.attrs['ome'] = {
            'version': '0.5',
            'multiscales': [{
                'axes': [{'name': 't', 'type': 'time'}, {'name': 'c', 'type': 'channel'},
                         {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                         {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                         {'name': 'x', 'type': 'space', 'unit': 'micrometer'}],
                'datasets': [{'path': 'scale0/image', 'coordinateTransformations': [
                    {'type': 'scale', 'scale': [1.0, 1.0, 1.0, PIXEL, PIXEL]},
                    {'type': 'translation', 'translation': [0.0, 0.0, 0.0, ty, tx]}]}],
            }],
        }
        zarr.consolidate_metadata(root.store)
print(f'wrote {GRID * GRID} single-resolution stores of {SIZE}x{SIZE} to {OUT}')
