import numpy as np
import xarray as xr

from src.register import save_image


if __name__ == '__main__':
    filename = 'output/test.ome.tiff'
    data = xr.DataArray(
        np.zeros((1, 16, 16), dtype=np.uint8),
        dims=list('cyx'),
        coords={'c': ['test_channel']},
        attrs={'channels': {'label': 'test_channel', 'color': [0, 1, 0]}},
    )
    #data.assign_coords({'c': ['test_channel']})
    save_image(filename, data, transform_key='registered')
