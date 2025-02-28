import logging
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os

from src.image.ome_tiff_helper import load_tiff, save_tiff
from src.image.util import float2int_image, int2float_image, create_quantile_images


def flatfield_correction(sims, transform_key, foreground_map, flatfield_quantile):
    norm_image_filename = f'resources/norm{flatfield_quantile}.tiff'
    if os.path.exists(norm_image_filename):
        logging.warning('Loading cached normalisation image')
        max_image = load_tiff(norm_image_filename)
    else:
        max_image = calc_flatfield_image(sims, foreground_map, flatfield_quantile)
        save_tiff(norm_image_filename, max_image)
    return apply_flatfield_correction(max_image, sims, transform_key)

def calc_flatfield_image(sims, foreground_map, flatfield_quantile):
    dtype = sims[0].dtype
    back_sims = [sim for sim, is_foreground in zip(sims, foreground_map) if not is_foreground]
    norm_images = create_quantile_images(back_sims, quantiles=[flatfield_quantile])
    max_image = norm_images[0]
    maxval = 2 ** (8 * dtype.itemsize) - 1
    max_image = max_image / np.float32(maxval)
    return max_image

def apply_flatfield_correction(flatfield_image, sims, transform_key):
    new_sims = []
    dtype = sims[0].dtype
    for sim in sims:
        image = float2int_image(image_flatfield_correction(int2float_image(sim), bright=flatfield_image), dtype)
        new_sim = si_utils.get_sim_from_array(
            image,
            dims=sim.dims,
            scale=si_utils.get_spacing_from_sim(sim),
            translation=si_utils.get_origin_from_sim(sim),
            transform_key=transform_key,
            affine=si_utils.get_affine_from_sim(sim, transform_key),
            c_coords=sim.c
        )
        new_sims.append(new_sim)
    return new_sims

def image_flatfield_correction(image0, dark=0, bright=1, clip=True):
    # Input/output: float images
    # https://en.wikipedia.org/wiki/Flat-field_correction
    mean_bright_dark = np.mean(bright - dark, (0, 1))
    image = (image0 - dark) * mean_bright_dark / (bright - dark)
    if clip:
        image = np.clip(image, 0, 1)
    return image
