from tifffile import TiffWriter, PHOTOMETRIC

from src.util import *


def save_ome_tiff(filename, data, pixel_size, channels=[], positions=[],
                  tile_size=(256, 256), compression='LZW', scaler=None):

    if data.ndim <= 3 and data.shape[-1] in (3, 4):
        photometric = PHOTOMETRIC.RGB
        move_channel = False
    else:
        photometric = PHOTOMETRIC.MINISBLACK
        # move channel axis to front
        move_channel = (data.ndim >= 3 and data.shape[-1] < data.shape[0])

    ome_metadata, resolution0, resolution_unit0 = create_tiff_metadata(pixel_size, positions, channels, is_ome=True)

    if scaler is not None:
        npyramid_add = scaler.max_layer
    else:
        npyramid_add = 0

    chunking0 = tuple(reversed(tile_size))
    chunking = tuple(reversed(tile_size))
    with TiffWriter(filename) as writer:
        for i in range(npyramid_add + 1):
            if i == 0:
                subifds = npyramid_add
                subfiletype = None
                metadata = ome_metadata
                resolution = resolution0
                resolutionunit = resolution_unit0
            else:
                subifds = None
                subfiletype = 1
                metadata = None
                resolution = None
                resolutionunit = None
                data = scaler.resize_image(data)
            chunking = retuple(chunking, data.shape)
            data = data.rechunk(chunks=chunking)

            ordered_data = np.moveaxis(data, -1, 0) if move_channel else data

            writer.write(ordered_data, photometric=photometric, subifds=subifds, subfiletype=subfiletype,
                         tile=chunking0, compression=compression,
                         resolution=resolution, resolutionunit=resolutionunit, metadata=metadata)


def create_tiff_metadata(pixel_size, positions=[], channels=[], is_ome=False):
    ome_metadata = None
    resolution = None
    resolution_unit = None
    pixel_size_um = None

    if pixel_size is not None:
        pixel_size_um = get_value_units_micrometer(pixel_size)
        resolution_unit = 'CENTIMETER'
        resolution = [1e4 / size for size in pixel_size_um]

    if is_ome:
        ome_metadata = {'Creator': 'multiview-stitcher'}
        ome_channels = []
        if pixel_size_um is not None:
            ome_metadata['PhysicalSizeX'] = float(pixel_size_um[0])
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeY'] = float(pixel_size_um[1])
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
            if len(pixel_size_um) > 2:
                ome_metadata['PhysicalSizeZ'] = float(pixel_size_um[2])
                ome_metadata['PhysicalSizeZUnit'] = 'µm'
        if positions is not None and len(positions) > 0:
            plane_metadata = {}
            plane_metadata['PositionX'] = [float(position[0]) for position in positions]
            plane_metadata['PositionXUnit'] = ['µm' for _ in positions]
            plane_metadata['PositionY'] = [float(position[1]) for position in positions]
            plane_metadata['PositionYUnit'] = ['µm' for _ in positions]
            if len(positions[0]) > 2:
                plane_metadata['PositionZ'] = [float(position[2]) for position in positions]
                plane_metadata['PositionZUnit'] = ['µm' for _ in positions]
            ome_metadata['Plane'] = plane_metadata
        for channeli, channel in enumerate(channels):
            ome_channel = {'Name': channel.get('label', str(channeli))}
            if 'color' in channel:
                ome_channel['Color'] = rgba_to_int(channel['color'])
            ome_channels.append(ome_channel)
        if ome_channels:
            ome_metadata['Channel'] = ome_channels
    return ome_metadata, resolution, resolution_unit
