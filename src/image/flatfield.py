from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os

from src.image.ome_tiff_helper import load_tiff, save_tiff
from src.image.util import float2int_image, int2float_image, create_quantile_images


def flatfield_correction(sims, transform_key, quantiles, foreground_map=None, cache_location=None):
    quantile_images = []
    if cache_location is not None:
        for quantile in quantiles:
            filename = cache_location + f'quantile{quantile}.tiff'
            if os.path.exists(filename):
                quantile_images.append(load_tiff(filename))

    if len(quantile_images) < len(quantiles):
        quantile_images = calc_flatfield_images(sims, quantiles, foreground_map)
        if cache_location is not None:
            for quantile, quantile_image in zip(quantiles, quantile_images):
                filename = cache_location + f'quantile{quantile}.tiff'
                save_tiff(filename, quantile_image)

    return apply_flatfield_correction(sims, transform_key, quantiles, quantile_images)

def calc_flatfield_images(sims, quantiles, foreground_map=None):
    if foreground_map is not None:
        back_sims = [sim for sim, is_foreground in zip(sims, foreground_map) if not is_foreground]
    else:
        back_sims = sims
    dtype = sims[0].dtype
    maxval = 2 ** (8 * dtype.itemsize) - 1
    flatfield_images = [image / np.float32(maxval) for image in create_quantile_images(back_sims, quantiles=quantiles)]
    return flatfield_images

def apply_flatfield_correction(sims, transform_key, quantiles, quantile_images):
    new_sims = []
    dtype = sims[0].dtype
    dark = 0
    bright = 1
    for quantile, quantile_image in zip(quantiles, quantile_images):
        if quantile < 0.5:
            dark = quantile_image
        else:
            bright = quantile_image
    for sim in sims:
        image = float2int_image(image_flatfield_correction(int2float_image(sim), dark=dark, bright=bright), dtype)
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
