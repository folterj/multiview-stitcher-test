from dask.diagnostics import ProgressBar
import glob
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from multiview_stitcher import registration, fusion, msi_utils, vis_utils
from multiview_stitcher import spatial_image_utils as si_utils

from src.TiffSource import TiffSource

mpl.rcParams['figure.dpi'] = 150


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


def load_tiles(file_pattern):
    tiles = []
    for file in glob.glob(file_pattern):
        source = TiffSource(file)
        tile = {'dim_order': source.get_dimension_order(),
                'translation': source.position,
                'scale': source.get_pixel_size_micrometer(),
                'channels': source.get_channels(),
                'data': source.get_source_dask()}
        tiles.append(tile)
    return tiles


def images_to_msims(tiles):
    # build input for stitching
    msims = []
    # convert string to list of characters

    for tile in tiles:
        # input data (can be any numpy compatible array: numpy, dask, cupy, etc.)
        sim = si_utils.get_sim_from_array(
            tile['data'],
            dims=list(tile['dim_order']),
            scale=tile['scale'],
            translation=tile['translation'],
            transform_key="stage_metadata",
            c_coords=tile['channels'],
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
    return msims


def register(msims, reg_channel=None):
    with ProgressBar():
        mappings = registration.register(
            msims,
            reg_channel=reg_channel,
            transform_key="stage_metadata",
            new_transform_key="translation_registered",
        )

    fused_image = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key="translation_registered",
    )
    return mappings, fused_image


def show_image(image, title='', cmap=None):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if cmap is None:
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #tiles = create_example_tiles()
    #reg_channel = "DAPI"

    file_pattern = 'D:/slides/EM04768_01_substrate_04/Reflection/20_percent_overlap/subselection/*.ome.tif'
    tiles = load_tiles(file_pattern)

    msims = images_to_msims(tiles)

    # plot the tile configuration
    vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False)

    #mappings, fused_image = register(msims, reg_channel)
    mappings, fused_image = register(msims)

    for mapping in mappings:
        print(mapping.data)
        print()

    # plot the tile configuration after registration
    vis_utils.plot_positions(msims, transform_key='translation_registered', use_positional_colors=False)

    # get fused array as a dask array
    show_image(fused_image.data[0, 0, 5, ...])
