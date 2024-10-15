import dask.array as da
import xarray as xr

from src.register import save_image


if __name__ == '__main__':
    filename = 'output/test.ome.tiff'
    channels = [{'label': 'Reflection', 'color': (1, 1, 1)},
                {'label': 'Fluorescence', 'color': (0, 1, 0)}]

    data = xr.DataArray(
        da.random.randint(0, 255, size=(2, 2048, 2048), dtype=da.uint16),
        dims=list('cyx'),
        coords={'c': [channel['label'] for channel in channels]},
        attrs={'channels': channels},
    )
    #data.assign_coords({'c': ['test_channel']})
    #data.assign_attrs({'channels': channels})

    save_image(filename, data, channels=channels)
