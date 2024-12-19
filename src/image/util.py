import cv2 as cv
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean
from tifffile import TiffFile
try:
    import matplotlib as mpl
    #mpl.rcParams['backend'] = 'svg'
    mpl.rcParams['figure.dpi'] = 300
    import matplotlib.pyplot as plt
except Exception as e:
    print(f'matplotlib import error:\n{e}')

from src.util import *


def show_image(image: np.ndarray):
    plt.imshow(image)
    plt.show()


def show_image_gray(image: np.ndarray):
    plt.imshow(image, cmap='gray')
    plt.show()


def int2float_image(image):
    source_dtype = image.dtype
    if not source_dtype.kind == 'f':
        maxval = 2 ** (8 * source_dtype.itemsize) - 1
        return image / np.float32(maxval)
    else:
        return image


def float2int_image(image, target_dtype=np.dtype(np.uint8)):
    source_dtype = image.dtype
    if source_dtype.kind not in ('i', 'u') and not target_dtype.kind == 'f':
        maxval = 2 ** (8 * target_dtype.itemsize) - 1
        return (image * maxval).astype(target_dtype)
    else:
        return image


def ensure_unsigned_type(dtype: np.dtype) -> np.dtype:
    new_dtype = dtype
    if dtype.kind == 'i' or dtype.byteorder == '>' or dtype.byteorder == '<':
        new_dtype = np.dtype(f'u{dtype.itemsize}')
    return new_dtype


def ensure_unsigned_image(image: np.ndarray) -> np.ndarray:
    source_dtype = image.dtype
    dtype = ensure_unsigned_type(source_dtype)
    if dtype != source_dtype:
        # conversion without overhead
        offset = 2 ** (8 * dtype.itemsize - 1)
        new_image = image.astype(dtype) + offset
    else:
        new_image = image
    return new_image


def convert_image_sign_type(image: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    source_dtype = image.dtype
    if source_dtype.kind == target_dtype.kind:
        new_image = image
    elif source_dtype.kind == 'i':
        new_image = ensure_unsigned_image(image)
    else:
        # conversion without overhead
        offset = 2 ** (8 * target_dtype.itemsize - 1)
        new_image = (image - offset).astype(target_dtype)
    return new_image


def redimension_data(data, old_order, new_order, **indices):
    # able to provide optional dimension values e.g. t=0, z=0
    if new_order == old_order:
        return data

    new_data = data
    order = old_order
    # remove
    for o in old_order:
        if o not in new_order:
            index = order.index(o)
            dim_value = indices.get(o, 0)
            new_data = np.take(new_data, indices=dim_value, axis=index)
            order = order[:index] + order[index + 1:]
    # add
    for o in new_order:
        if o not in order:
            new_data = np.expand_dims(new_data, 0)
            order = o + order
    # move
    old_indices = [order.index(o) for o in new_order]
    new_indices = list(range(len(new_order)))
    new_data = np.moveaxis(new_data, old_indices, new_indices)
    return new_data


def get_numpy_slicing(dimension_order, **slicing):
    slices = []
    for axis in dimension_order:
        index = slicing.get(axis)
        index0 = slicing.get(axis + '0')
        index1 = slicing.get(axis + '1')
        if index0 is not None and index1 is not None:
            slice1 = slice(int(index0), int(index1))
        elif index is not None:
            slice1 = int(index)
        else:
            slice1 = slice(None)
        slices.append(slice1)
    return tuple(slices)


def get_image_quantile(image: np.ndarray, quantile: float, axis=None) -> float:
    value = np.quantile(image, quantile, axis=axis).astype(image.dtype)
    return value


def normalise_values(image: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return np.clip((image.astype(np.float32) - min_value) / (max_value - min_value), 0, 1)


def norm_image_variance(image0):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = np.clip((image - np.mean(image)) / np.std(image), 0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def norm_image_quantiles(image0, quantile=0.99):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    min_value = np.quantile(image, 1 - quantile)
    max_value = np.quantile(image, quantile)
    normimage = np.clip((image - min_value) / (max_value - min_value), 0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def get_image_size_info(sizes_xyzct: list, pixel_nbytes: int, pixel_type: np.dtype, channels: list) -> str:
    image_size_info = 'XYZCT:'
    size = 0
    for i, size_xyzct in enumerate(sizes_xyzct):
        w, h, zs, cs, ts = size_xyzct
        size += np.int64(pixel_nbytes) * w * h * zs * cs * ts
        if i > 0:
            image_size_info += ','
        image_size_info += f' {w} {h} {zs} {cs} {ts}'
    image_size_info += f' Pixel type: {pixel_type} Uncompressed: {print_hbytes(size)}'
    if sizes_xyzct[0][3] == 3:
        channel_info = 'rgb'
    else:
        channel_info = ','.join([channel.get('Name', '') for channel in channels])
    if channel_info != '':
        image_size_info += f' Channels: {channel_info}'
    return image_size_info


def pilmode_to_pixelinfo(mode: str) -> tuple:
    pixelinfo = (np.uint8, 8, 1)
    mode_types = {
        'I': (np.uint32, 32, 1),
        'F': (np.float32, 32, 1),
        'RGB': (np.uint8, 24, 3),
        'RGBA': (np.uint8, 32, 4),
        'CMYK': (np.uint8, 32, 4),
        'YCbCr': (np.uint8, 24, 3),
        'LAB': (np.uint8, 24, 3),
        'HSV': (np.uint8, 24, 3),
    }
    if '16' in mode:
        pixelinfo = (np.uint16, 16, 1)
    elif '32' in mode:
        pixelinfo = (np.uint32, 32, 1)
    elif mode in mode_types:
        pixelinfo = mode_types[mode]
    pixelinfo = (np.dtype(pixelinfo[0]), pixelinfo[1])
    return pixelinfo


def calc_pyramid(xyzct: tuple, npyramid_add: int = 0, pyramid_downsample: float = 2,
                 volumetric_resize: bool = False) -> list:
    x, y, z, c, t = xyzct
    if volumetric_resize and z > 1:
        size = (x, y, z)
    else:
        size = (x, y)
    sizes_add = []
    scale = 1
    for _ in range(npyramid_add):
        scale /= pyramid_downsample
        scaled_size = np.maximum(np.round(np.multiply(size, scale)).astype(int), 1)
        sizes_add.append(scaled_size)
    return sizes_add


def image_reshape(image: np.ndarray, target_size: tuple) -> np.ndarray:
    tw, th = target_size
    sh, sw = image.shape[0:2]
    if sw < tw or sh < th:
        dw = max(tw - sw, 0)
        dh = max(th - sh, 0)
        padding = [(0, dh), (0, dw)]
        if len(image.shape) == 3:
            padding += [(0, 0)]
        image = np.pad(image, padding, 'edge')
    if tw < sw or th < sh:
        image = image[0:th, 0:tw]
    return image


def resize_image(image, new_size):
    if not isinstance(new_size, (tuple, list, np.ndarray)):
        # use single value for width; apply aspect ratio
        size = np.flip(image.shape[:2])
        new_size = new_size, new_size * size[1] // size[0]
    return cv.resize(image, new_size)


def image_resize(image: np.ndarray, target_size0: tuple, dimension_order: str = 'yxc') -> np.ndarray:
    shape = image.shape
    x_index = dimension_order.index('x')
    y_index = dimension_order.index('y')
    c_is_at_end = ('c' in dimension_order and dimension_order.endswith('c'))
    size = shape[x_index], shape[y_index]
    if np.mean(np.divide(size, target_size0)) < 1:
        interpolation = cv.INTER_CUBIC
    else:
        interpolation = cv.INTER_AREA
    dtype0 = image.dtype
    image = ensure_unsigned_image(image)
    target_size = tuple(np.maximum(np.round(target_size0).astype(int), 1))
    if dimension_order in ['yxc', 'yx']:
        new_image = cv.resize(np.asarray(image), target_size, interpolation=interpolation)
    elif dimension_order == 'cyx':
        new_image = np.moveaxis(image, 0, -1)
        new_image = cv.resize(np.asarray(new_image), target_size, interpolation=interpolation)
        new_image = np.moveaxis(new_image, -1, 0)
    else:
        ts = image.shape[dimension_order.index('t')] if 't' in dimension_order else 1
        zs = image.shape[dimension_order.index('z')] if 'z' in dimension_order else 1
        target_shape = list(image.shape).copy()
        target_shape[x_index] = target_size[0]
        target_shape[y_index] = target_size[1]
        new_image = np.zeros(target_shape, dtype=image.dtype)
        for t in range(ts):
            for z in range(zs):
                slices = get_numpy_slicing(dimension_order, z=z, t=t)
                image1 = image[slices]
                if not c_is_at_end:
                    image1 = np.moveaxis(image1, 0, -1)
                new_image1 = np.atleast_3d(cv.resize(np.asarray(image1), target_size, interpolation=interpolation))
                if not c_is_at_end:
                    new_image1 = np.moveaxis(new_image1, -1, 0)
                new_image[slices] = new_image1
    new_image = convert_image_sign_type(new_image, dtype0)
    return new_image


def precise_resize(image: np.ndarray, factors) -> np.ndarray:
    if image.ndim > len(factors):
        factors = list(factors) + [1]
    new_image = downscale_local_mean(np.asarray(image), tuple(factors)).astype(image.dtype)
    return new_image


def create_compression_filter(compression: list) -> tuple:
    compressor, compression_filters = None, None
    compression = ensure_list(compression)
    if compression is not None and len(compression) > 0:
        compression_type = compression[0].lower()
        if len(compression) > 1:
            level = int(compression[1])
        else:
            level = None
        if 'lzw' in compression_type:
            from imagecodecs.numcodecs import Lzw
            compression_filters = [Lzw()]
        elif '2k' in compression_type or '2000' in compression_type:
            from imagecodecs.numcodecs import Jpeg2k
            compression_filters = [Jpeg2k(level=level)]
        elif 'jpegls' in compression_type:
            from imagecodecs.numcodecs import Jpegls
            compression_filters = [Jpegls(level=level)]
        elif 'jpegxr' in compression_type:
            from imagecodecs.numcodecs import Jpegxr
            compression_filters = [Jpegxr(level=level)]
        elif 'jpegxl' in compression_type:
            from imagecodecs.numcodecs import Jpegxl
            compression_filters = [Jpegxl(level=level)]
        else:
            compressor = compression
    return compressor, compression_filters


def create_compression_codecs(compression: list) -> list:
    codecs = None
    compression = ensure_list(compression)
    if compression is not None and len(compression) > 0:
        compression_type = compression[0].lower()
        if len(compression) > 1:
            level = int(compression[1])
        else:
            level = None
        if 'lzw' in compression_type:
            from imagecodecs.numcodecs import Lzw
            codecs = [Lzw()]
        elif '2k' in compression_type or '2000' in compression_type:
            from imagecodecs.numcodecs import Jpeg2k
            codecs = [Jpeg2k(level=level)]
        elif 'jpegls' in compression_type:
            from imagecodecs.numcodecs import Jpegls
            codecs = [Jpegls(level=level)]
        elif 'jpegxr' in compression_type:
            from imagecodecs.numcodecs import Jpegxr
            codecs = [Jpegxr(level=level)]
        elif 'jpegxl' in compression_type:
            from imagecodecs.numcodecs import Jpegxl
            codecs = [Jpegxl(level=level)]
        else:
            codecs = [compression]
    return codecs


def get_tiff_pages(tiff: TiffFile) -> list:
    pages = []
    found = False
    if tiff.series and not tiff.is_mmstack:
        # has series
        baseline = tiff.series[0]
        for level in baseline.levels:
            # has levels
            level_pages = []
            for page in level.pages:
                found = True
                level_pages.append(page)
            if level_pages:
                pages.append(level_pages)

    if not found:
        for page in tiff.pages:
            pages.append(page)
    return pages


def tags_to_dict(tags: tifffile.TiffTags) -> dict:
    tag_dict = {}
    for tag in tags.values():
        tag_dict[tag.name] = tag.value
    return tag_dict


def tiff_info(filename: str) -> str:
    s = ''
    nom_size = 0
    tiff = TiffFile(filename)
    real_size = tiff.fstat.st_size
    s += str(tiff) + '\n'
    if tiff.ome_metadata:
        print(tiff.ome_metadata)
        s += f'OME: {print_dict(tifffile.xml2dict(tiff.ome_metadata))}\n'
    if tiff.metaseries_metadata:
        s += f'Series: {tiff.metaseries_metadata}\n'
    if tiff.imagej_metadata:
        s += f'ImageJ: {tiff.imagej_metadata}\n'

    for page0 in get_tiff_pages(tiff):
        page = page0[0] if isinstance(page0, list) else page0
        s += str(page) + '\n'
        s += f'Size: {np.flip(page.shape)} ({print_hbytes(page.size)})\n'
        if page.is_tiled:
            s += f'Tiling: {page.tilewidth} {page.tilelength} {page.tiledepth}\n'
        s += f'Compression: {str(page.compression)} jpegtables: {page.jpegtables is not None}\n'
        tag_dict = tags_to_dict(page.tags)
        if 'TileOffsets' in tag_dict:
            tag_dict.pop('TileOffsets')
        if 'TileByteCounts' in tag_dict:
            tag_dict.pop('TileByteCounts')
        if 'ImageDescription' in tag_dict and tag_dict['ImageDescription'].startswith('<?xml'):
            # redundant
            tag_dict.pop('ImageDescription')
        s += print_dict(tag_dict) + '\n\n'
        nom_size += page.size

    s += f'Overall compression: 1:{nom_size / real_size:.1f}'
    return s


def tiff_info_short(filename: str) -> str:
    nom_size = 0
    tiff = TiffFile(filename)
    s = str(filename)
    real_size = tiff.fstat.st_size
    for page in tiff.pages:
        s += ' ' + str(page)
        nom_size += page.size
    s += f' Image size:{nom_size} File size:{real_size} Overall compression: 1:{nom_size / real_size:.1f}'
    return s


def compare_image(image0, image1, show=False) -> float:
    dif, dif_max, dif_mean, psnr = compare_image_dist(image0, image1)
    print(f'rgb dist max: {dif_max:.1f} mean: {dif_mean:.1f} PSNR: {psnr:.1f}')
    if show:
        show_image(dif)
        show_image((dif * 10).astype(np.uint8))
    return dif


def compare_image_dist(image0: np.ndarray, image1: np.ndarray) -> tuple:
    dif = cv.absdiff(image0, image1)
    psnr = cv.PSNR(image0, image1)
    if dif.size > 1000000000:
        # split very large array
        rgb_maxs = []
        rgb_means = []
        for dif1 in np.array_split(dif, 16):
            rgb_dif = np.linalg.norm(dif1, axis=2)
            rgb_maxs.append(np.max(rgb_dif))
            rgb_means.append(np.mean(rgb_dif))
        rgb_max = np.max(rgb_maxs)
        rgb_mean = np.mean(rgb_means)
    else:
        rgb_dif = np.linalg.norm(dif, axis=2)
        rgb_max = np.max(rgb_dif)
        rgb_mean = np.mean(rgb_dif)
    return dif, rgb_max, rgb_mean, psnr


def compare_image_dist_simple(image0: np.ndarray, image1: np.ndarray) -> dict:
    dif = cv.absdiff(image0, image1)
    psnr = cv.PSNR(image0, image1)
    rgb_dif = np.linalg.norm(dif, axis=2)
    dif_max = np.max(rgb_dif)
    dif_mean = np.mean(rgb_dif)
    return {'dif_max': dif_max, 'dif_mean': dif_mean, 'psnr': psnr}


def calc_fraction_used(image: np.ndarray, threshold: float = 0.1) -> float:
    low = int(round(threshold * 255))
    high = int(round((1 - threshold) * 255))
    shape = image.shape
    total = shape[0] * shape[1]
    good = 0
    for y in range(shape[0]):
        for x in range(shape[1]):
            pixel = image[y, x]
            if low <= pixel[0] < high and low <= pixel[1] < high and low <= pixel[2] < high:
                good += 1
    fraction = good / total
    return fraction


def blur_image_single(image, sigma):
    return gaussian_filter(image, sigma)


def blur_image(image, sigma):
    nchannels = image.shape[2] if image.ndim == 3 else 1
    if nchannels not in [1, 3]:
        new_image = np.zeros_like(image)
        for channeli in range(nchannels):
            new_image[..., channeli] = blur_image_single(image[..., channeli], sigma)
    else:
        new_image = blur_image_single(image, sigma)
    return new_image


def calc_images_median(images):
    out_image = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
    median_image = np.median(images, 0, out_image)
    return median_image


def calc_images_quantiles(images, quantiles):
    quantile_images = [image.astype(np.float32) for image in np.quantile(images, quantiles, 0)]
    return quantile_images


def create_normalisation_images(images, quantiles, nchannels=1):
    channel_images2 = []
    for channeli in range(nchannels):
        if nchannels > 1:
            channel_images = [image[..., channeli] for image in images]
        else:
            channel_images = images
        # Filter tiles with signal
        median_image = calc_images_median(channel_images)
        difs = [np.mean(np.abs(image.astype(np.float32) - median_image.astype(np.float32)), (0, 1)) for image in images]
        threshold = np.mean(difs, 0)
        #threshold, _ = cv.threshold(np.array(difs).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
        #threshold, foregrounds = filter_noise_images(channel_images)
        #back_images = [image for image, foreground in zip(images, foregrounds) if not foreground]
        back_images = [image for image, dif in zip(images, difs) if np.all(dif < threshold)]
        norm_images = calc_images_quantiles(back_images, quantiles)
        channel_images2.append(norm_images)

    quantile_images = []
    for quantilei in range(len(quantiles)):
        quantile_image = None
        for channel_image in channel_images2:
            image = channel_image[quantilei]
            if quantile_image is None:
                quantile_image = image
            else:
                quantile_image = cv.merge(list(cv.split(quantile_image)) + [image])
        quantile_images.append(quantile_image)
    return quantile_images


def flatfield_correction(image0, dark=0, bright=1, clip=True):
    # Input/output: float images
    # https://en.wikipedia.org/wiki/Flat-field_correction
    mean_bright_dark = np.mean(bright - dark, (0, 1))
    image = (image0 - dark) * mean_bright_dark / (bright - dark)
    if clip:
        image = np.clip(image, 0, 1)
    return image


def get_max_downsamples(shape, npyramid_add, pyramid_downsample):
    shape = list(shape)
    for i in range(npyramid_add):
        shape[-1] //= pyramid_downsample
        shape[-2] //= pyramid_downsample
        if shape[-1] < 1 or shape[-2] < 1:
            return i
    return npyramid_add


def filter_noise_images(images):
    dtype = images[0].dtype
    maxval = 2 ** (8 * dtype.itemsize) - 1
    image_vars = [np.asarray(np.std(image)).item() for image in images]
    threshold, mask0 = cv.threshold(np.array(image_vars).astype(dtype), 0, maxval, cv.THRESH_OTSU)
    mask = [flag.item() for flag in mask0.astype(bool)]
    return int(threshold), mask


def detect_area_points(image):
    method = cv.THRESH_OTSU
    threshold = -5
    contours = []
    while len(contours) <= 1 and threshold <= 255:
        _, binimage = cv.threshold(image, threshold, 255, method)
        contours0 = cv.findContours(binimage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours0[0] if len(contours0) == 2 else contours0[1]
        method = cv.THRESH_BINARY
        threshold += 5
    area_contours = [(contour, cv.contourArea(contour)) for contour in contours]
    area_contours.sort(key=lambda contour_area: contour_area[1], reverse=True)
    min_area = max(np.mean([area for contour, area in area_contours]), 1)
    area_points = [(get_center(contour), area) for contour, area in area_contours if area > min_area]

    #image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    #for point in area_points:
    #    radius = int(np.round(np.sqrt(point[1]/np.pi)))
    #    cv.circle(image, tuple(np.round(point[0]).astype(int)), radius, (255, 0, 0), -1)
    #show_image(image)
    return area_points


def points_to_3d(points):
    return [list(point) + [0] for point in points]
