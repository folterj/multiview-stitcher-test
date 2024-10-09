from ome_zarr.scale import Scaler
from tifffile import TiffWriter, PHOTOMETRIC

from src.util import *


def save_ome_tiff(filename, data, pixel_size, position, channels,
                  tile_size=(256, 256), compression='LZW',
                  npyramid_add=0, pyramid_downsample=2):
    if data.ndim <= 3 and data.shape[-1] in (3, 4):
        photometric = PHOTOMETRIC.RGB
        move_channel = False
    else:
        photometric = PHOTOMETRIC.MINISBLACK
        # move channel axis to front
        move_channel = (data.ndim >= 3 and data.shape[-1] < data.shape[0])

    ome_metadata, resolution, resolution_unit = create_tiff_metadata(pixel_size, position, channels, is_ome=True)

    scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)

    with TiffWriter(filename) as writer:
        ordered_data = np.moveaxis(data, -1, 0) if move_channel else data
        writer.write(ordered_data, photometric=photometric, subifds=npyramid_add,
                     tile=tile_size, compression=compression,
                     resolution=resolution, resolutionunit=resolution_unit, metadata=ome_metadata)
        for i in range(npyramid_add):
            data = scaler.resize_image(data)
            ordered_data = np.moveaxis(data, -1, 0) if move_channel else data
            writer.write(ordered_data, subfiletype=1,
                         tile=tile_size, compression=compression)


def create_tiff_metadata(pixel_size, position, channels, is_ome=False):
    ome_metadata = None
    resolution = None
    resolution_unit = None
    pixel_size_um = None

    if pixel_size is not None:
        pixel_size_um = get_value_units_micrometer(pixel_size)
        resolution_unit = 'CENTIMETER'
        resolution = [1e4 / size for size in pixel_size_um]

    if is_ome:
        ome_metadata = {}
        ome_channels = []
        if pixel_size_um is not None:
            ome_metadata['PhysicalSizeX'] = pixel_size_um[0]
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeY'] = pixel_size_um[1]
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
            if len(pixel_size_um) > 2:
                ome_metadata['PhysicalSizeZ'] = pixel_size_um[2]
                ome_metadata['PhysicalSizeZUnit'] = 'µm'
        if position is not None:
            plane_metadata = {}
            plane_metadata['PositionX'] = position[0]
            plane_metadata['PositionXUnit'] = 'µm'
            plane_metadata['PositionY'] = position[1]
            plane_metadata['PositionYUnit'] = 'µm'
            if len(position) > 2:
                ome_metadata['PositionZ'] = position[2]
                ome_metadata['PositionZUnit'] = 'µm'
            ome_metadata['Plane'] = plane_metadata
        for channel in channels:
            ome_channel = {'Name': channel.get('label', '')}
            if 'color' in channel:
                ome_channel['Color'] = rgba_to_int(channel['color'])
            ome_channels.append(ome_channel)
        if ome_channels:
            ome_metadata['Channel'] = ome_channels
    return ome_metadata, resolution, resolution_unit
