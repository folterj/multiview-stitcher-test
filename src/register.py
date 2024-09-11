from dask.diagnostics import ProgressBar
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
from multiview_stitcher import registration, fusion, msi_utils, vis_utils
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os
import re
from spatial_image import SpatialImage
from tqdm import tqdm

from src.OmeZarr import OmeZarr
from src.TiffSource import TiffSource
from src.util import *


mpl.rcParams['figure.dpi'] = 300


def create_example_tiles():
    tile_translations = [
        {"z": 2.5, "y": -10, "x": 30},
        {"z": 2.5, "y": 30, "x": 10},
        {"z": 2.5, "y": 30, "x": 50},
    ]
    dim_order = "czyx"
    scale = {"z": 2, "y": 0.5, "x": 0.5}
    channels = ["DAPI", "GFP"]

    tiles = [{'dim_order': dim_order,
              'translation': tile_translation,
              'scale': scale,
              'channels': channels,
              'data': np.random.randint(0, 100, (2, 10, 100, 100))}
             for tile_translation in tile_translations]
    return tiles


def init_tiles(files):
    tiles = []
    for file in tqdm(files):
        source = TiffSource(file)
        translation = convert_xyz_to_dict(get_value_units_micrometer(source.position))
        scale = convert_xyz_to_dict(source.get_pixel_size_micrometer())
        if not translation.keys() == scale.keys():
            translation = {key: translation[key] for key in scale.keys()}
        data = source.get_source_dask()[0]
        # rotate 180 degrees
        data = data[::-1, ::-1]
        tile = {'dim_order': source.dimension_order,
                'translation': translation,
                'scale': scale,
                'channels': source.get_channels(),
                'data': data}
        tiles.append(tile)
    return tiles


def images_to_msims(tiles):
    # build input for stitching
    msims = []
    for tile in tqdm(tiles):
        # input data (can be any numpy compatible array: numpy, dask, cupy, etc.)
        channel_labels = [channel.get('label', '') for channel in tile['channels']]
        sim = si_utils.get_sim_from_array(
            tile['data'],
            dims=list(tile['dim_order']),
            scale=tile['scale'],
            translation=tile['translation'],
            transform_key="stage_metadata",
            c_coords=channel_labels
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
    return msims


def register(msims, reg_channel=None, reg_channel_index=None):
    print('Registering...')
    if isinstance(reg_channel, int):
        reg_channel_index = reg_channel
        reg_channel = None
    with ProgressBar():
        mappings = registration.register(
            msims,
            reg_channel=reg_channel,
            reg_channel_index=reg_channel_index,
            transform_key="stage_metadata",
            new_transform_key="translation_registered"
        )

    print('Fusing...')
    fused_image = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key="translation_registered"
    )
    return mappings, fused_image


def save_zarr(filename, image, source):
    #data.to_zarr(filename)
    if isinstance(image, SpatialImage):
        source.output_dimension_order = ''.join(image.dims)
    zarr = OmeZarr(filename)
    zarr.write(image.data, source)


def convert_xyz_to_dict(xyz):
    dct = {dim: value for dim, value in zip('xyz', xyz)}
    return dct


def dir_regex(pattern):
    dir = os.path.dirname(pattern)
    file_pattern = os.path.basename(pattern)
    files = [os.path.join(dir, file) for file in os.listdir(dir) if re.search(file_pattern, file)]
    files_sorted = sorted(files, key=lambda file: find_all_numbers(get_filetitle(file)))
    return files_sorted


def show_image(image, title='', cmap=None):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if cmap is None:
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.tight_layout()
    plt.show()


def run():
    #tiles = create_example_tiles()
    #reg_channel = "DAPI"

    #file_pattern = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/subselection/tiles_1_MMStack_New Grid 1-Grid_(?!0_0.ome.tif).*'
    #file_pattern = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/tiles_1_MMStack_New Grid 1-Grid_0_.*.ome.tif'
    #file_pattern = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    file_pattern = '/nemo/project/proj-czi-vp/raw/lm/EM04768_01_substrate_04/Reflection/20_percent_overlap/ome_tif_reflection/converted/.*.ome.tif'
    reg_channel = 0

    output_dir = 'output'

    original_tiles_filename = os.path.join(output_dir, 'tiles_original.png')
    original_fused_filename = os.path.join(output_dir, 'original.ome.zarr')
    registered_tiles_filename = os.path.join(output_dir, 'tiles_registered.png')
    registered_fused_filename = os.path.join(output_dir, 'registered.ome.zarr')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mvsr_logger = logging.getLogger('multiview_stitcher.registration')
    mvsr_logger.setLevel(logging.INFO)
    if len(mvsr_logger.handlers) == 0:
        mvsr_logger.addHandler(logging.StreamHandler())

    print('Initialising tiles...')
    filenames = dir_regex(file_pattern)
    file_indices = ['-'.join(map(str, find_all_numbers(get_filetitle(filename))[-2:])) for filename in filenames]
    source0 = TiffSource(filenames[0])
    tiles = init_tiles(filenames)

    print('Converting tiles...')
    msims = images_to_msims(tiles)

    # before registration:
    print('Fusing original...')
    original_fused = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key='stage_metadata'
    )

    # plot the tile configuration
    print('Plotting tiles...')
    vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=False, output_filename=original_tiles_filename)

    print('Saving fused image...')
    save_zarr(original_fused_filename, original_fused, source0)
    #show_image(original_fused.data[0, 0, ...])

    mappings, registered_fused = register(msims, reg_channel)

    print('Registration mappings (delta position offsets for each tile):')
    for mapping in mappings:
        print(mapping.data)
        print()

    # plot the tile configuration after registration
    print('Plotting tiles...')
    vis_utils.plot_positions(msims, transform_key='translation_registered', use_positional_colors=False,
                             view_labels=file_indices, view_labels_size=3,
                             show_plot=False, output_filename=registered_tiles_filename)

    print('Saving fused image...')
    save_zarr(registered_fused_filename, registered_fused, source0)
    #show_image(registered_fused.data[0, 0, 5, ...]) # XYZ example data - show middle of Z depth
    #show_image(registered_fused.data[0, 0, ...])


if __name__ == '__main__':
    run()
