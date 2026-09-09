import itertools
import logging
from math import ceil
import time

import cv2 as cv
import dask
import dask.array as da
import numpy as np
from multiview_stitcher import msi_utils, param_utils, fusion, mv_graph
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.registration import _get_overlap_bboxes, sims_to_intrinsic_coord_system, \
    get_affine_from_intrinsic_affine
from scipy.spatial import ConvexHull
from skimage.filters import gaussian
from skimage.feature import blob_log, plot_matched_features
from skimage.transform import downscale_local_mean, resize
import xarray as xr
from xarray import DataTree

try:
    import matplotlib as mpl
    mpl.use('Agg')
    #mpl.rcParams['backend'] = 'svg'
    mpl.rcParams['figure.dpi'] = 300
    import matplotlib.pyplot as plt
except Exception as e:
    print(f'matplotlib import error:\n{e}')

from muvis_align.constants import (default_chunk_size, default_contrast_limits_max_tasks,
                                   default_fusion_chunk_bytes, default_preview_max_bytes,
                                   fusion_stack_arrays)
from muvis_align.util import *


def show_image(image, title='', cmap=None):
    if cmap is None:
        nchannels = image.shape[2] if len(image.shape) > 2 else 1
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.show()


def plt_close():
    plt.close()


def grayscale_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 4:
        return cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    elif nchannels > 1:
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        return image


def _adapt_transform_to_image_dims(sim, transform, transform_key):
    """
    Extract only the spatial dimensions from transform that match the image.
    
    For 2D images, extracts the 3x3 submatrix from a 4x4 transform.
    Handles dimension mismatch between transforms and images.
    """
    sim_spatial_dims = si_utils.get_spatial_dims_from_sim(sim)
    transform = sim.attrs['transforms'][transform_key]
    
    # Get dimensions from transform coordinates
    transform_spatial_dims = list(transform.coords['x_in'].values)
    
    # Count actual spatial dims (exclude '1' padding)
    transform_spatial_dim_count = len([d for d in transform_spatial_dims if d != '1'])
    sim_spatial_dim_count = len(sim_spatial_dims)
    
    if transform_spatial_dim_count == sim_spatial_dim_count:
        # Transform already matches - no adaptation needed
        return transform
    
    # Find which spatial dims are in both sim and transform (ignoring '1' padding)
    relevant_dims = [d for d in sim_spatial_dims if d in transform_spatial_dims]
    
    # Extract submatrix for only relevant dimensions (+ 1 for homogeneous coordinate)
    relevant_dim_names = relevant_dims + ['1']
    adapted = transform.sel(x_in=relevant_dim_names, x_out=relevant_dim_names)
    
    return adapted


def color_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 1:
        return cv.cvtColor(np.array(image), cv.COLOR_GRAY2RGB)
    else:
        return image


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


def uint8_image(image):
    source_dtype = image.dtype
    if source_dtype.kind == 'f':
        image = image * 255
    elif source_dtype.itemsize != 1:
        factor = 2 ** (8 * (source_dtype.itemsize - 1))
        image = image // factor
    return image.astype(np.uint8)


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


def redimension_sim_data(image, old_order, new_order, **indices):
    # xarray-native equivalent of redimension_data: lazy .isel()/.expand_dims()/.transpose() on an
    # existing (already dask-backed) DataArray, instead of numpy ops on a raw array - keeps whatever
    # coords the dims already have (e.g. 'c' channel labels, 't' timepoints), only touching dims
    # old_order/new_order actually mention. Presence/absence is always checked against image.dims
    # itself (never a separately-tracked old_order/new_order string) - image can already carry
    # extra dims neither order lists (e.g. a forced 't' when redimensioning a 2D source into a 3D
    # output_order), and those are left untouched rather than assumed absent, which would otherwise
    # make the final transpose miss a dim it doesn't know exists.
    if new_order == old_order and set(new_order) == set(image.dims):
        return image

    for o in old_order:
        if o not in new_order and o in image.dims:
            dim_value = indices.get(o, 0)
            image = image.isel({o: dim_value})
    for o in new_order:
        if o not in image.dims:
            image = image.expand_dims(o, axis=0)
    # any dim still on image but not in new_order (e.g. that untouched forced 't') is carried
    # through as-is, ahead of the requested order
    extra_dims = [dim for dim in image.dims if dim not in new_order]
    return image.transpose(*extra_dims, *list(new_order))


def ensure_spatial_image_dims(image, c_coords=None, t_coords=None):
    # si_utils.get_sim_from_array unconditionally forces 'c' and 't' dims to be present (size 1 if
    # not already there) on every sim it builds - this is a hard multiview_stitcher convention that
    # downstream code (channel detection, fusion, chunking) relies on. Match it exactly so a sim
    # extracted from a msim built without going through get_sim_from_array is indistinguishable.
    if c_coords is None and 'c' in image.coords:
        c_coords = np.atleast_1d(image.coords['c'].values)
    if t_coords is None and 't' in image.coords:
        t_coords = np.atleast_1d(image.coords['t'].values)
    for dim, coords in (('c', c_coords), ('t', t_coords)):
        if dim not in image.dims:
            # a stale scalar (non-dimension) coord of the same name can't survive expand_dims as-is
            image = image.reset_coords(dim, drop=True) if dim in image.coords else image
            image = image.expand_dims(dim, axis=0)
            if coords is not None:
                image = image.assign_coords({dim: list(coords)})
    new_dims = [dim for dim in si_utils.SPATIAL_IMAGE_DIMS if dim in image.dims]
    if list(new_dims) != list(image.dims):
        image = image.transpose(*new_dims)
    return image


def rechunk_if_monolithic(image, chunk_size):
    # a badly-chunked source (e.g. a single-chunk TIFF) needs splitting into smaller dask chunks
    # for reasonable memory/parallelism downstream - a no-op if it's already chunked
    if chunk_size and len(image.chunksizes.get('x', ())) == 1 and len(image.chunksizes.get('y', ())) == 1:
        image = image.chunk(xyz_to_dict([chunk_size] * 2 if isinstance(chunk_size, int) else chunk_size))
    return image


def calc_pyramid_level_factors(sizes, pyramid_downsample=2, min_size=default_chunk_size):
    """Cumulative per-dim downsample factor for each pyramid level needed *beyond* the one whose
    spatial extents are `sizes` ({dim: size}), halving while the largest extent still exceeds
    min_size - i.e. until a level small enough to draw a zoomed-out overview from exists.

    The single definition of "how many coarser levels does this need, and how much coarser":
    build_missing_pyramid_levels() applies it to synthesize those levels on the reader side, and
    ome_zarr_helper.get_padding_scale_factors() to write them on the export side, so a written
    file carries exactly the levels a reader would otherwise have had to invent. Sizes are
    rounded up as they shrink, matching the strided slicing build_missing_pyramid_levels() uses
    (data[::2] of 4097 rows is 2049, not 2048).
    """
    factors = []
    if not sizes:
        return factors
    current = dict(sizes)
    cumulative = {dim: 1 for dim in sizes}
    while max(current.values()) > min_size:
        step = {dim: (pyramid_downsample if size >= pyramid_downsample else 1)
                for dim, size in current.items()}
        if all(value == 1 for value in step.values()):
            break
        current = {dim: ceil(size / step[dim]) for dim, size in current.items()}
        cumulative = {dim: factor * step[dim] for dim, factor in cumulative.items()}
        factors.append(dict(cumulative))
    return factors


def build_missing_pyramid_levels(data, dimension_order, pixel_size, pyramid_downsample=2,
                                 min_size=default_chunk_size):
    """A source with only one real resolution (e.g. a plain, non-pyramidal TIFF) leaves napari
    with no coarse level to show while zoomed out, so drawing it forces computing the *entire*
    finest-level dask graph just to render a thumbnail-sized view - the usual cause of a slow
    first draw despite loading (building the lazy graph) itself being fast. Synthesize coarser
    levels so a small one always exists for that overview; full resolution is only ever computed
    once the user actually zooms in that far.

    Deliberately strided (nearest-neighbour) subsampling, not a real mean-downsample: a
    non-pyramidal source is typically also a single monolithic dask chunk (an untiled TIFF
    strip/page, decoded whole regardless of what slice is asked of it) - `data` itself has
    already paid that one decode. A mean-downsample chain would then run len(levels)-1 extra
    full-array reduction passes on top of that decode just to get a thumbnail only ever used for
    a quick, zoomed-out preview; slicing every `pyramid_downsample`-th pixel instead reuses the
    same already-decoded array for near-zero extra cost. Measured on a real, non-pyramidal 47MP
    EM tile: dropped get_contrast_limits() (which reads this coarsest level) from ~0.46s back to
    ~0.07s, i.e. down to roughly the cost of the one unavoidable decode.

    How many levels, and how much coarser each is, comes from calc_pyramid_level_factors() -
    shared with the export side (see there), so a synthesized pyramid and a written one agree.

    Returns ([data] + extra levels, [pixel_size] + matching per-level pixel sizes) - a no-op
    (single-level) result for a source with no spatial dims at all.
    """
    spatial_axes = {dim: axis for axis, dim in enumerate(dimension_order) if dim in 'xyz'}
    datas = [data]
    pixel_sizes = [pixel_size]
    if not spatial_axes:
        return datas, pixel_sizes
    sizes = {dim: data.shape[axis] for dim, axis in spatial_axes.items()}
    for factors in calc_pyramid_level_factors(sizes, pyramid_downsample, min_size):
        # factors are cumulative against the finest level, so always slice `data` itself rather
        # than the previous level - one strided view of the already-decoded array either way
        slicing = tuple(slice(None, None, factors.get(dim, 1))
                        for dim in dimension_order)
        coarse = data[slicing]
        if coarse.shape == datas[-1].shape:
            break
        datas.append(coarse)
        pixel_sizes.append({dim: pixel_size[dim] * data.shape[spatial_axes[dim]] / coarse.shape[spatial_axes[dim]]
                            for dim in pixel_size if dim in spatial_axes})
    return datas, pixel_sizes


def build_source_redimensioned_msim(source, output_order, chunk_size=default_chunk_size):
    """Redimension `source.msim`'s own per-level 'image' DataArrays into `output_order` (lazy
    transpose/expand_dims), ensure the 'c'/'t' dims every sim needs, and rechunk any level still
    monolithic in x/y. Depends only on `source` and `output_order`, never on per-run geometry
    (translation/transform) - build_source_msim() calls this through source.get_msim(), which
    caches the result, rather than redoing this work on every call. Does not mutate source.msim -
    a fresh DataTree is built and returned.
    """
    # every sim's 'c' dim is forced to exist (size 1 if the source is single-channel) by
    # si_utils.get_sim_from_array - label it unconditionally so a channel selected by name
    # (e.g. registration's 'channel' param) can be found via .sel(c=...) regardless of whether
    # the source is natively multi-channel
    c_coords = [channel.get('label', '') for channel in source.get_channels()]
    datasets = {}
    for scale_key in msi_utils.get_sorted_scale_keys(source.msim):
        image = source.msim[scale_key].ds['image']
        image = redimension_sim_data(image, source.dimension_order, output_order)
        image = ensure_spatial_image_dims(image, c_coords=c_coords)
        image = rechunk_if_monolithic(image, chunk_size)
        datasets[scale_key] = xr.Dataset({'image': image})
    return DataTree.from_dict(datasets)


def build_source_msim(source, output_order, translation, transform, transform_key, z_scale=None):
    """
    Build a new msim for `source` covering every real pyramid level, redimensioned to `output_order`
    and re-geometried with `translation` (intrinsic coords, per level) + `transform` (extrinsic affine,
    same at every level) - starting from source.get_msim(output_order) (redimensioned once, then
    cached on `source`) rather than reconstructing from raw arrays via si_utils.get_sim_from_array.
    Does not mutate the cached msim - a fresh DataTree is built and returned.
    """
    if transform is None:
        spatial_dims = [dim for dim in output_order if dim in 'xyz']
        xaffine = param_utils.identity_transform(len(spatial_dims))
    else:
        xaffine = param_utils.affine_to_xaffine(transform)

    translation_arg = dict(translation)
    if translation_arg:
        if 'x' not in translation_arg:
            translation_arg['x'] = 0
        if 'y' not in translation_arg:
            translation_arg['y'] = 0

    redimensioned_msim = source.get_msim(output_order)
    scale_keys = msi_utils.get_sorted_scale_keys(redimensioned_msim)
    datasets = {}
    for level, scale_key in enumerate(scale_keys):
        image = redimensioned_msim[scale_key].ds['image']

        pixel_size = dict(source.pixel_sizes[level])
        if 'z' in output_order and 'z' not in pixel_size:
            pixel_size['z'] = abs(z_scale) if z_scale else 1
        spatial_dims = si_utils.get_spatial_dims_from_sim(image)
        new_coords = {dim: translation_arg.get(dim, 0) + np.arange(image.sizes[dim]) * pixel_size.get(dim, 1)
                      for dim in spatial_dims}
        image = image.assign_coords(new_coords)

        ds = xr.Dataset({'image': image})
        ds[transform_key] = xaffine
        datasets[scale_key] = ds

    return DataTree.from_dict(datasets)


def build_source_stack_props(source, output_order, translation, transform, transform_key,
                             z_scale=None, level=0, promote_z=False):
    """The geometry the shapes/overlap-shapes preview needs - shape, spacing, origin and
    transform - derived straight from a source's metadata, allocating nothing.

    This is all si_utils.get_stack_properties_from_sim() would have read back off a sim, so
    building one first (and, with it, an array for the sim to wrap) is pure ceremony for a path
    that never touches a pixel: drawing bounding boxes should not need image data to exist, let
    alone be read. build_source_shape_sim() below still produces that sim for the one caller
    that genuinely needs one - multiview_stitcher's exact overlap test takes sims - and is
    itself built on this, so the two can never describe different geometry.

    Mirrors build_source_shape_sim()'s own conventions exactly (there is a test asserting so):
    a dim of output_order the source lacks becomes size 1; x/y default to origin 0 as soon as
    any translation is given; and promote_z adds a size-1 'z' at the source's own z position,
    widening the transform to 3D with it.
    """
    spatial_order = [dim for dim in si_utils.SPATIAL_IMAGE_DIMS if dim in 'zyx'
                     and (dim in output_order or (promote_z and dim == 'z'))]
    sizes = dict(zip(source.dimension_order, source.get_shape(level)))
    shape = {dim: int(sizes.get(dim, 1)) for dim in spatial_order}

    pixel_size = dict(source.pixel_sizes[level])
    if 'z' in spatial_order and 'z' not in pixel_size:
        pixel_size['z'] = abs(z_scale) if z_scale else 1
    # si_utils derives spacing from the differences between coordinates, which a size-1 dim has
    # none of - it reports 1.0 there whatever the nominal pixel size, so match that rather than
    # the value the coords were built from
    spacing = {dim: float(pixel_size.get(dim, 1)) if shape[dim] > 1 else 1.0
               for dim in spatial_order}

    translation_arg = dict(translation)
    if translation_arg:
        translation_arg.setdefault('x', 0)
        translation_arg.setdefault('y', 0)
    origin = {dim: float(translation_arg.get(dim, 0)) for dim in spatial_order}

    if transform is None:
        xaffine = param_utils.identity_transform(len([dim for dim in output_order if dim in 'xyz']))
    else:
        xaffine = param_utils.affine_to_xaffine(transform)
    if promote_z:
        xaffine = widen_xaffine_to_3d(xaffine)

    stack_props = {'shape': shape, 'spacing': spacing, 'origin': origin}
    if transform_key is not None:
        stack_props['transform'] = xaffine
    return stack_props


def build_source_shape_sim(source, output_order, translation, transform, transform_key, z_scale=None, level=0,
                           promote_z=False):
    """Cheap, single-level equivalent of build_source_msim(), for the one consumer that needs an
    actual sim rather than plain geometry: multiview_stitcher's exact (linprog) overlap test
    takes sims. Everything else in the shapes path works off build_source_stack_props() instead,
    which allocates nothing - see there.

    Built from those same stack properties, so the two cannot describe different geometry, and
    wrapping a lazily-allocated placeholder of the right shape and dtype rather than the
    source's own array: nothing here reads a pixel, while source.data/get_level_data() would
    open the file (for OME-Zarr, build the whole msim), which is exactly the work source
    initialisation defers - see ImageSource.data.

    promote_z=True mirrors make_msims_3d()'s own promotion (see promote_sim_to_3d()) for the
    case where output_order itself has no 'z' (every source is individually 2D) but different
    sources sit at different z heights (e.g. a z-stack of 2D tiles) - without this, a source's
    own z position/translation would otherwise be silently dropped instead of becoming a real,
    if size-1, 'z' dim/coordinate on the returned sim.
    """
    stack_props = build_source_stack_props(source, output_order, translation, transform,
                                           transform_key, z_scale=z_scale, level=level,
                                           promote_z=promote_z)
    c_coords = [channel.get('label', '') for channel in source.get_channels()]
    return sim_from_stack_props(stack_props, source.dtype, transform_key, c_coords=c_coords)


def sim_from_stack_props(stack_props, dtype, transform_key, c_coords=None):
    """A sim carrying exactly the geometry in `stack_props` and no image data of its own - for
    the multiview_stitcher entry points that take sims but only ever read geometry off them.

    The array behind it is a lazy one-chunk placeholder: it has to carry a shape and a dtype,
    nothing more. Deliberately a dask array and not, say, a zero-strided numpy view over a
    single element - that is far cheaper to create, but xarray then materialises it into a real
    array (measured at 90ms for a 6400x6400 tile) where it leaves a lazy one alone. name=False
    skips dask's deterministic tokenization, pointless for a placeholder nothing looks up.
    """
    spatial_dims = list(stack_props['shape'])
    shape = tuple(stack_props['shape'][dim] for dim in spatial_dims)
    data = da.zeros(shape, dtype=dtype, chunks=shape, name=False)
    image = si_utils.get_sim_from_array(
        data, dims=spatial_dims,
        scale=stack_props['spacing'], translation=stack_props['origin'],
        transform_key=transform_key, c_coords=c_coords)
    if 'transform' in stack_props:
        si_utils.set_sim_affine(image, stack_props['transform'], transform_key)
    return image


def map_msim_levels(msim, level_func):
    """Apply level_func(sim, scale_key) -> new_sim independently to every scale level of `msim`,
    reassembling the results into a new multiscale msim covering the same levels. Each level is
    extracted/rebuilt through the ordinary sim <-> msim helpers (msi_utils.get_sim_from_msim /
    get_msim_from_sims) - the per-level results are genuinely new sims (e.g. gaussian-filtered,
    normalised), not a restamp of existing data, so there's no coordinate-reuse trap to avoid here.
    """
    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    level_sims = [level_func(msi_utils.get_sim_from_msim(msim, scale=scale_key), scale_key)
                  for scale_key in scale_keys]
    return msi_utils.get_msim_from_sims(level_sims)


def get_msim_level_data(msim):
    """Raw dask array per pyramid level, straight off each scale node's own Dataset (ImageSource.
    get_level_data's pattern, generalised to every level at once) - no sim wrapping needed, since
    napari's multiscale add_image only ever wants the arrays themselves.
    """
    return [msim[scale_key].ds['image'].data for scale_key in msi_utils.get_sorted_scale_keys(msim)]


def get_chunk_sizes(dtype, spatial_dims, num_sources=1, num_z_positions=1,
                    xy_chunk_size=default_chunk_size, target_bytes=64 * 1024 ** 2,
                    fusion_target_bytes=None, min_xy_chunk_size=64):
    """Per-spatial-dim chunk sizes for a fused preview, budgeted against what fusing one output
    chunk actually costs in memory - which is set by how many *sources* land in that chunk, not
    by the chunk's own size on disk.

    multiview_stitcher's fusion works one output chunk at a time, but for each chunk it
    transforms every overlapping source into a full-chunk-sized float32 array and np.stack()s
    them (see fusion._core: field_ims_t, plus a same-shaped blending-weight stack and their
    product - fusion_stack_arrays of them). Peak memory for one chunk is therefore
    ~views_in_chunk * chunk_voxels * 4 bytes * fusion_stack_arrays, entirely independent of the
    output dtype.
    Sizing chunks by output bytes alone (target_bytes / itemsize) misses that factor completely:
    for a few thousand sources it lands on chunks whose fusion needs hundreds of GB.

    Two things make coarse pyramid levels the worst case, not the finest:

    - fuse() reuses one output_chunksize for *every* level it builds, so an xy chunk size larger
      than a coarse level's whole extent collapses that level into a single chunk covering the
      full field of view - and therefore every source in it. Cost is worst at the level whose
      extent is about xy_chunk_size (coarser than that the chunk shrinks with the level; finer,
      the chunk stays fixed while the sources per chunk fall), so bound that level.
    - when sources are distributed along z (num_z_positions > 1: a stack of 2D sections, each
      its own set of tiles), a chunk spanning Nz output planes pulls in every source from Nz
      sections at once. That multiplies views per chunk *and* chunk voxels, so cost grows with
      the square of the z chunk - one plane per chunk is what keeps a zoomed-out view cheap.

    fusion_target_bytes is the budget for *one* chunk, defaulting to
    default_fusion_chunk_bytes, which is derived from this job's own CPU and memory allocation:
    napari (and any dask.compute over a whole level) fuses one chunk per worker concurrently, so
    process peak is roughly the budget times the worker count. That is what lets the same sizing
    serve a 64-core/2TB HPC node (which lands on the generous xy_chunk_size, keeping the chunk
    count - and so the graph-construction cost - low) and a laptop (which trades chunk size for
    staying inside its own memory) without either being tuned for the other.

    x/y stay as generous as that budget allows (napari always shows both axes in full, so
    fragmenting them buys nothing, and x/y sizes are rounded down to a multiple of
    min_xy_chunk_size to keep the chunk grid tidy), and z takes the remainder: with sources
    spread over z it is driven down to 1, and only for a genuine z-stack (num_z_positions == 1,
    where a deeper chunk adds voxels but no extra views) does it fall back to filling
    target_bytes - keeping z chunks small enough that viewing one slice doesn't force computing
    many slices' worth of fusion. An isotropic split (the same size on every axis, independent
    of which one is actually sliced through) knows neither distinction.
    """
    # sources that a single output plane can draw from - the per-z-plane tile count when
    # sources are spread over z, otherwise every source (they all sit at the same height)
    sources_per_plane = max(1, round(num_sources / max(1, num_z_positions)))
    # budget expressed in float32 voxels, across the stacked arrays fusion holds at once
    if fusion_target_bytes is None:
        fusion_target_bytes = default_fusion_chunk_bytes
    voxel_budget = max(1.0, fusion_target_bytes / (4 * fusion_stack_arrays))

    def fit_xy(views_per_chunk):
        # largest x/y chunk whose fusion stays within budget, rounded down to a whole number of
        # min_xy_chunk_size blocks (never below one such block - a smaller chunk grid than that
        # costs more in dask tasks and graph build than it saves in peak memory)
        size = min(xy_chunk_size, np.sqrt(voxel_budget / views_per_chunk))
        return max(min_xy_chunk_size, int(size // min_xy_chunk_size) * min_xy_chunk_size)

    if 'z' not in spatial_dims:
        # 2D output: one chunk's cost is sources_per_plane * xy_size ** 2
        xy_size = fit_xy(sources_per_plane)
        return {dim: xy_size for dim in spatial_dims if dim in ('x', 'y')}

    xy_size = fit_xy(sources_per_plane)
    sizes = {dim: xy_size for dim in spatial_dims if dim in ('x', 'y')}
    if num_z_positions > 1:
        # a z chunk of Nz sections costs sources_per_plane * Nz ** 2 * xy_size ** 2 voxels (Nz
        # times the sources, Nz times the voxels) - quadratic in Nz, so once sources sit at
        # distinct z positions the whole budget goes to x/y and z stays at a single plane
        sizes['z'] = 1
    else:
        # genuine z-stack: extra z depth adds voxels but no extra views, so the output-byte
        # budget stays the binding constraint, capped by the fusion budget for this many sources.
        # Floor, not round: rounding up trades a budget overrun for a chunk depth nobody asked
        # for, and z is the axis a deeper chunk hurts most (napari computes a whole chunk to
        # show one slice)
        voxels_per_chunk = min(target_bytes / np.dtype(dtype).itemsize,
                               voxel_budget / sources_per_plane)
        sizes['z'] = max(1, int(voxels_per_chunk // xy_size ** 2))
    return sizes


def get_contrast_limits(msim, cheap=False, max_tasks=default_contrast_limits_max_tasks):
    """Real min/max contrast range computed from just the coarsest pyramid level, so a caller
    can pass it as add_image()'s contrast_limits without napari falling back to its own default:
    for multiscale layers that already reads the coarsest level (data[-1]), but for anything
    other than uint8 still computes a real min/max over it, which for a still-lazy dask array
    means eagerly running that level's whole fusion graph just to pick initial display bounds
    (see napari.layers.utils.layer_utils.calc_data_range). Doing it here instead is no cheaper
    per se, but runs once, up front, on only the coarsest (by far the smallest) level.

    cheap=True skips that compute entirely and returns a naive dtype-range guess instead - for
    an interactive overview built from hundreds of sources, one dask.compute() per source adds
    up fast even on just the coarsest level; the user can always auto-contrast a layer from
    napari's own UI once it's up, so getting this exactly right up front isn't worth the cost
    there.

    That naive guess is also the automatic fallback whenever the coarsest level's own graph is
    larger than max_tasks: "the coarsest level is small" holds for its pixel count, but not
    necessarily for the work behind it - a level fused from thousands of sources is thousands of
    transforms however few pixels come out. This step exists to be the fast one before anything
    is on screen, so past that size it declines to be the thing that blocks first paint.
    """
    coarsest = get_msim_level_data(msim)[-1]
    if not cheap:
        num_tasks = len(coarsest.dask) if hasattr(coarsest, 'dask') else 0
        if num_tasks > max_tasks:
            logging.info(f'Contrast limits: using dtype range instead of computing'
                         f' the coarsest pyramid level ({num_tasks} tasks > {max_tasks})')
            cheap = True
    if cheap:
        dtype = coarsest.dtype
        if np.issubdtype(dtype, np.integer):
            return [0, np.iinfo(dtype).max]
        return [0.0, 1.0]
    min_val, max_val = dask.compute(coarsest.min(), coarsest.max())
    min_val, max_val = float(min_val), float(max_val)
    if min_val == max_val:
        max_val = min_val + 1
    return [min_val, max_val]


def get_msim_image0(msim, level=0):
    """The raw 'image' DataArray at a given pyramid level (default: the finest, scale0) - straight
    off that scale node's own Dataset, no sim wrapping. si_utils.get_spacing_from_sim/
    get_origin_from_sim/get_spatial_dims_from_sim, and plain .dims/.sizes/.dtype checks, all work
    identically on this as they would on a full sim (they only ever read .dims/.coords) - a sim
    is only needed once something reads .attrs['transforms'] (si_utils.get_affine_from_sim et al).
    """
    scale_key = msi_utils.get_sorted_scale_keys(msim)[level]
    return msim[scale_key].ds['image']


def get_msim_dims(msim):
    # .dims is identical whether read off the full sim (get_sim_from_msim) or the scale node's
    # own 'image' DataArray directly - no need to build a sim just to compare dims
    return get_msim_image0(msim).dims


def get_msim_transform_keys(msim):
    # every transform key is its own data_var (alongside 'image') on each scale node's Dataset -
    # reading them directly answers "does this msim have transform_key set yet" without building
    # a sim just to call si_utils.get_tranform_keys_from_sim
    scale_key = msi_utils.get_sorted_scale_keys(msim)[0]
    return set(msim[scale_key].ds.data_vars.keys()) - {'image'}


def extract_sims_from_fused(result):
    """Extract a concrete sim (or list of sims) from a fuse() result, which is always msims: a
    single fused multiscale msim (DataTree), or, in 'compose' mode (no actual fusion), a list of
    per-source msims. Used only at the boundary where a downstream consumer genuinely needs a sim
    (save_image() has no multiscale support)."""
    if isinstance(result, list):
        return [msi_utils.get_sim_from_msim(msim, scale=msi_utils.get_sorted_scale_keys(msim)[0]) for msim in result]
    return msi_utils.get_sim_from_msim(result, scale=msi_utils.get_sorted_scale_keys(result)[0])


def wrap_sims_as_msims(sims):
    # trivial single-level msim per sim - the escape hatch for callers that only have a concrete
    # sim (an ad-hoc resolution with no corresponding real pyramid, e.g. a resized preview) but
    # still need to satisfy a msims-only API (fuse()); scale_factors=[] avoids computing a
    # synthetic downsample pyramid nothing here would use
    return [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]


def combine_msims_as_channels(msims, channel_labels):
    """Combine several single-channel msims (one per source) into one multichannel msim, stacking
    along a new 'c' dim - per pyramid level, so the result stays a genuine multiscale msim rather
    than collapsing to a single resolution. `msims` here are each expected to already share the
    same geometry/levels (e.g. the per-source results of fusing each source individually against
    the same output_stack_properties)."""
    scale_keys = msi_utils.get_sorted_scale_keys(msims[0])
    level_sims = []
    for scale_key in scale_keys:
        channel_sims = [msi_utils.get_sim_from_msim(msim, scale=scale_key) for msim in msims]
        channel_sims = [sim.assign_coords({'c': [label]}) for sim, label in zip(channel_sims, channel_labels)]
        combined = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
        level_sims.append(combined)
    return msi_utils.get_msim_from_sims(level_sims)


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


def get_level_from_scale(source, target_scale=1):
    # Only downscaling
    if isinstance(target_scale, dict):
        # dict of desired pixel size
        target_pixel_size = target_scale
        target_scale = {dim: target_pixel_size[dim] / source_pixel_size
                        for dim, source_pixel_size in source.get_pixel_size().items()}
    elif isinstance(target_scale, str):
        # target pixel size with unit
        index = target_scale.find(next(filter(str.isalpha, target_scale)))
        pixel_size = convert_to_um(float(target_scale[:index]), target_scale[index:])
        target_pixel_size = {dim: pixel_size for dim in source.get_pixel_size()}
        target_scale = {dim: pixel_size / source_pixel_size
                        for dim, source_pixel_size in source.get_pixel_size().items()}
    else:
        # target scale factor
        target_pixel_size = {dim: float(source_pixel_size * target_scale)
                             for dim, source_pixel_size in source.get_pixel_size().items()}
        target_scale = {dim: target_scale for dim in source.get_pixel_size()}
    best_level, best_scale = 0, target_scale
    for level, factors in enumerate(source.scale_factors):
        if any(np.isclose(factors[dim], target_scale[dim], rtol=1e-4) for dim in factors):
            best_level, best_scale = level, {dim: target_scale[dim] / factors[dim] for dim in factors}
            break
        if any(factors[dim] <= target_scale[dim] for dim in factors):
            best_level, best_scale = level, {dim: target_scale[dim] / factors[dim] for dim in factors}
    if best_level == 0:
        for dim in best_scale:
            if best_scale[dim] < 1:
                best_scale[dim] = 1
    return best_level, best_scale, target_pixel_size


def image_reshape(image: np.ndarray, target_size: tuple) -> np.ndarray:
    tw, th = target_size
    sh, sw = image.shape[0:2]
    if sw < tw or sh < th:
        dw = max(tw - sw, 0)
        dh = max(th - sh, 0)
        padding = [(dh // 2, dh - dh //  2), (dw // 2, dw - dw // 2)]
        if len(image.shape) == 3:
            padding += [(0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=(0, 0))
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


def draw_keypoints(image, points, color=(255, 0, 0)):
    out_image = color_image(float2int_image(image))
    for point in points:
        point = np.round(point).astype(int)
        cv.drawMarker(out_image, tuple(point), color=color, markerType=cv.MARKER_CROSS, markerSize=5, thickness=1)
    return out_image


def draw_keypoints_matches_cv(image1, points1, image2, points2, matches=None, inliers=None,
                              color=(255, 0, 0), inlier_color=(0, 255, 0), radius = 15, thickness = 2):
    # based on https://gist.github.com/woolpeeker/d7e1821e1b5c556b32aafe10b7a1b7e8
    image1 = uint8_image(image1)
    image2 = uint8_image(image2)
    # We're drawing them side by side.  Get dimensions accordingly.
    new_shape = (max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3)
    out_image = np.zeros(new_shape, image1.dtype)
    # Place images onto the new image.
    out_image[0:image1.shape[0], 0:image1.shape[1]] = color_image(image1)
    out_image[0:image2.shape[0], image1.shape[1]:image1.shape[1] + image2.shape[1]] = color_image(image2)

    if matches is not None:
        # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
        for index, match in enumerate(matches):
            if inliers is not None and inliers[index]:
                line_color = inlier_color
            else:
                line_color = color
            # So the keypoint locs are stored as a tuple of floats.  cv2.line() wants locs as a tuple of ints.
            end1 = tuple(np.round(points1[match[0]]).astype(int))
            end2 = tuple(np.round(points2[match[1]]).astype(int) + np.array([image1.shape[1], 0]))
            cv.line(out_image, end1, end2, line_color, thickness)
            cv.circle(out_image, end1, radius, line_color, thickness)
            cv.circle(out_image, end2, radius, line_color, thickness)
    else:
        # Draw all points if no matches are provided.
        for point in points1:
            point = tuple(np.round(point).astype(int))
            cv.circle(out_image, point, radius, color, thickness)
        for point in points2:
            point = tuple(np.round(point).astype(int) + np.array([image1.shape[1], 0]))
            cv.circle(out_image, point, radius, color, thickness)
    return out_image


def draw_keypoints_matches_sk(image1, points1, image2, points2, matches=np.array([]),
                              show_plot=True, output_filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    shape_y, shape_x = image1.shape[:2]
    if shape_x > 2 * shape_y:
        alignment = 'vertical'
    else:
        alignment = 'horizontal'
    plot_matched_features(
        image1,
        image2,
        keypoints0=points1,
        keypoints1=points2,
        matches=matches,
        ax=ax,
        alignment=alignment,
        only_matches=True,
    )
    plt.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()


def draw_keypoints_matches(image1, points1, image2, points2, matches=[], inliers=[],
                           points_color='black', match_color='red', inlier_color='lime',
                           show_plot=True, output_filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    shape = np.max([image.shape for image in [image1, image2]], axis=0)
    shape_y, shape_x = shape[:2]
    if shape_x > 2 * shape_y:
        merge_axis = 0
        offset2 = [shape_y, 0]
    else:
        merge_axis = 1
        offset2 = [0, shape_x]
    image = np.concatenate([
        np.pad(image1, ((0, shape[0] - image1.shape[0]), (0, shape[1] - image1.shape[1]))),
        np.pad(image2, ((0, shape[0] - image2.shape[0]), (0, shape[1] - image2.shape[1])))
    ], axis=merge_axis)
    ax.imshow(image, cmap='gray')

    if len(points1) > 0:
        ax.scatter(
            points1[:, 1],
            points1[:, 0],
            facecolors='none',
            edgecolors=points_color,
        )
    if len(points2) > 0:
        ax.scatter(
            points2[:, 1] + offset2[1],
            points2[:, 0] + offset2[0],
            facecolors='none',
            edgecolors=points_color,
        )

    for i, match in enumerate(matches):
        color = match_color
        if i < len(inliers) and inliers[i]:
            color = inlier_color
        index1, index2 = match
        ax.plot(
            (points1[index1, 1], points2[index2, 1] + offset2[1]),
            (points1[index1, 0], points2[index2, 0] + offset2[0]),
            '-', linewidth=1, alpha=0.5, color=color,
        )

    plt.tight_layout()
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()

    return fig, ax


def draw_keypoints_matches_napari(image1, points1, image2, points2, matches=[], inliers=[],
                                  points_color='black', match_color='red', inlier_color='lime'):
    def _as_points_array(points):
        points = np.asarray(points)
        if points.size == 0:
            points = np.empty((0, 0), dtype=float)
        elif points.ndim == 1:
            points = points[None, :]
        return points.astype(float, copy=False)

    def _get_image_spatial_dims(image):
        image = np.asarray(image)
        if image.ndim <= 2:
            return image.ndim
        if image.shape[-1] in (3, 4):
            return image.ndim - 1
        return min(image.ndim, 3)

    points1 = _as_points_array(points1)
    points2 = _as_points_array(points2)
    images = [image for image in (image1, image2) if image is not None]

    point_dims = [pts.shape[1] for pts in (points1, points2) if len(pts) > 0]

    # Infer spatial dims from points when present, otherwise from images.
    if len(point_dims) > 0:
        spatial_dims = min(max(point_dims), 3)
    elif len(images) > 0:
        spatial_dims = max(_get_image_spatial_dims(image) for image in images)
    else:
        spatial_dims = 2

    # Infer spatial shape from images when present; otherwise from points.
    if len(images) > 0:
        shape = np.max([
            np.array(np.asarray(image).shape[:spatial_dims], dtype=int)
            for image in images
        ], axis=0)
    else:
        shape = np.ones(spatial_dims, dtype=int)
        all_points = [points[:, :spatial_dims] for points in (points1, points2)
                      if len(points) > 0 and points.shape[1] >= spatial_dims]
        if len(all_points) > 0:
            points_shape = np.ceil(np.max(np.vstack(all_points), axis=0)).astype(int) + 1
            shape = np.maximum(shape, points_shape)

    # Concatenate along the smallest spatial axis to keep the merged view compact.
    merge_axis = int(np.argmin(shape))
    offset2 = np.zeros(spatial_dims, dtype=float)
    offset2[merge_axis] = shape[merge_axis]

    # Pad each image to the same shape before concatenation; include non-spatial axes.
    max_ndim = max((image.ndim for image in images), default=spatial_dims)
    target_shape = np.ones(max_ndim, dtype=int)
    target_shape[:spatial_dims] = shape
    for image in images:
        ext_shape = np.array(image.shape + (1,) * (max_ndim - image.ndim), dtype=int)
        target_shape = np.maximum(target_shape, ext_shape)

    def _pad_image(image):
        if image is None:
            return np.zeros(tuple(target_shape), dtype=np.float32)
        image = np.asarray(image)
        if image.ndim < max_ndim:
            image = image.reshape(image.shape + (1,) * (max_ndim - image.ndim))
        padding = tuple((0, max(target_shape[axis] - image.shape[axis], 0)) for axis in range(max_ndim))
        return np.pad(image, padding)

    image = np.concatenate([_pad_image(image1), _pad_image(image2)], axis=merge_axis)

    # Build combined points in napari's expected coordinate order.
    p1 = points1[:, :spatial_dims] if (len(points1) > 0 and points1.shape[1] >= spatial_dims) else np.empty((0, spatial_dims))
    p2 = points2[:, :spatial_dims] + offset2 if (len(points2) > 0 and points2.shape[1] >= spatial_dims) else np.empty((0, spatial_dims))
    points_data = np.vstack([p1, p2]) if (len(p1) or len(p2)) else np.empty((0, spatial_dims))

    # Build match lines as shapes layers (each line is [[p1], [p2]]).
    non_inlier_line_data = []
    inlier_line_data = []
    for i, match in enumerate(matches):
        is_inlier = inliers[i] if i < len(inliers) else False
        i1, i2 = int(match[0]), int(match[1])
        valid_indices = i1 < len(points1) and i2 < len(points2)
        valid_dims = points1.shape[1] >= spatial_dims and points2.shape[1] >= spatial_dims
        if valid_indices and valid_dims:
            start = points1[i1, :spatial_dims]
            end = points2[i2, :spatial_dims] + offset2
            line = np.array([start, end], dtype=float)
            if is_inlier:
                inlier_line_data.append(line)
            else:
                non_inlier_line_data.append(line)

    layers = [
        (
            image,
            {
                "name": "matches_image",
                # For 2D grayscale this is ignored by napari; for RGB it is inferred.
                "rgb": (image.ndim >= 3 and image.shape[-1] in (3, 4)),
                "blending": "translucent_no_depth"
            },
            "image",
        )
    ]

    if len(points_data) > 0:
        layers.append(
            (
                points_data,
                {
                    "name": "keypoints",
                    "size": 6,
                    "face_color": points_color,
                    "border_color": "transparent",
                    "symbol": "ring",
                    "opacity": 0.5,
                },
                "points",
            )
        )

    if len(non_inlier_line_data) > 0:
        layers.append(
            (
                non_inlier_line_data,
                {
                    "name": "matches",
                    "shape_type": "line",
                    "edge_color": match_color,
                    "edge_width": 1,
                    "opacity": 0.25,
                },
                "shapes",
            )
        )

    if len(inlier_line_data) > 0:
        layers.append(
            (
                inlier_line_data,
                {
                    "name": "matches_inliers",
                    "shape_type": "line",
                    "edge_color": inlier_color,
                    "edge_width": 1,
                    "opacity": 0.25,
                },
                "shapes",
            )
        )

    return layers


def metric_to_rgb(value, min_light=0, max_light=1, output_range=1.0):
    # metric range 0...1 to red-yellow-green ranged rgb
    if value is None or np.isnan(value):
        return 0, 0, 0
    colormap = plt.colormaps.get('RdYlGn')
    index = int(value * colormap.N)
    r, g, b, a = [float(value) for value in colormap(index)]
    light = 0.2125 * r + 0.7154 * g + 0.0721 * b
    if light < min_light:
        factor = light / min_light
        r = 1 - (1 - r) * factor
        g = 1 - (1 - g) * factor
        b = 1 - (1 - b) * factor
    elif light > max_light:
        factor = max_light / light
        r *= factor
        g *= factor
        b *= factor
    r *= output_range
    g *= output_range
    b *= output_range
    if isinstance(output_range, int):
        r, g, b = int(r), int(g), int(b)
    return r, g, b


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


def gaussian_filter_image(image, sigma, is_3d=False):
    ndims = 4 if is_3d else 3
    nchannels = image.shape[-1] if image.ndim > ndims else 1
    if nchannels not in [1, 3]:
        new_image = np.zeros_like(image)
        for channeli in range(nchannels):
            new_image[..., channeli] = gaussian(image[..., channeli], sigma)
    else:
        new_image = gaussian(image, sigma, preserve_range=True)
    return new_image


def calc_images_median(images):
    out_image = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
    median_image = np.median(images, 0, out_image)
    return median_image


def calc_images_quantiles(images, quantiles):
    quantile_images = [image.astype(np.float32) for image in np.quantile(images, quantiles, 0)]
    return quantile_images


def get_image_quantile(image: np.ndarray, quantile: float, axis=None) -> float:
    # np.asarray() computes a dask-backed image via its __array__ protocol - a no-op for an
    # already-plain numpy image. Only ever called on a small, already-coarsest-level image, so
    # materializing it here avoids depending on dask's own quantile/percentile at all (e.g. dask
    # 2025.10 still passing numpy's removed interpolation= kwarg internally), rather than passing
    # a dask array straight into np.quantile() below and dispatching into that code path.
    image = np.asarray(image)
    if axis is None:
        image = image.ravel()
        axis = 0
    value = np.quantile(image, quantile, axis=axis).astype(image.dtype)
    return np.array(value).item()


def get_image_window(image, low=0.01, high=0.99):
    window = (
        get_image_quantile(image, low),
        get_image_quantile(image, high)
    )
    return window


def normalise_values(image: np.ndarray, min_value: float=None, max_value: float=None) -> np.ndarray:
    if min_value is None or max_value is None:
        min_value, max_value = get_image_window(image)
    image = (image.astype(np.float32) - min_value) / (max_value - min_value)
    return image.clip(0, 1)


def norm_image_variance(image0, is_3d=False):
    ncoldims = 4 if is_3d else 3
    if len(image0.shape) == ncoldims and image0.shape[-1] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = (image - np.mean(image)) / np.std(image)
    normimage = normimage.clip(0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def norm_image_variance2(image0, is_3d=False):
    ncoldims = 4 if is_3d else 3
    if len(image0.shape) == ncoldims and image0.shape[-1] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = ((image - np.mean(image)) / np.std(image) + 1) / 2
    normimage = normimage.clip(0, 1).astype(np.float32)
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
    normimage = (image - np.mean(image)) / (max_value - min_value)
    normimage = normimage.clip(0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


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


def invert_data(data):
    if isinstance(data, list):
        return [invert_data(d) for d in data]
    else:
        dtype = data.dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return info.max - data
        elif np.issubdtype(dtype, np.floating):
            return 1.0 - data
        else:
            return data


def detect_area_points(data):
    method = cv.THRESH_OTSU
    threshold = -5
    contours = []
    while len(contours) <= 1 and threshold <= 255:
        _, binimage = cv.threshold(np.array(uint8_image(data)), threshold, 255, method)
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


def detect_volume_points(data):
    blobs = blob_log(data, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1)
    if blobs.shape[1] > 3:
        blobs = blobs[:, :3]
    return blobs


def get_transforms(sims):
    # accepts either sims or msims - a msim's transform keys are read directly off its data_vars
    # (get_msim_transform_keys), no sim needs to be built at all just to enumerate them
    groups = [get_msim_transform_keys(item) if isinstance(item, DataTree)
             else si_utils.get_tranform_keys_from_sim(item)
             for item in sims]
    return list({a for group in groups for a in group})


def check_sim_dims(sim):
    origin = si_utils.get_origin_from_sim(sim)
    dims = {'dims': sim.dims,
            'origin': list(origin.keys())}
    for transform_key in si_utils.get_tranform_keys_from_sim(sim):
        transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        dims[transform_key] = np.array(transform.coords['x_in'])
    return dims


def copy_transforms(source_sims, target_sims, transform_key):
    # source_sims accepts either sims or msims - a msim's transform is read directly
    # (msi_utils.get_transform_from_msim), no sim round-trip needed (mirrors copy_transforms_to_msims)
    dims = list(si_utils.get_origin_from_sim(target_sims[0]).keys())
    for source_sim, target_sim in zip(source_sims, target_sims):
        if isinstance(source_sim, DataTree):
            transform = msi_utils.get_transform_from_msim(source_sim, transform_key)
        else:
            transform = si_utils.get_affine_from_sim(source_sim, transform_key=transform_key)
        transform_dims = np.array(transform.coords['x_in'])
        if len(transform_dims) - 1 != len(dims):
            new_transform = param_utils.identity_transform(ndim=len(dims))
            # Get common non-t dimensions for assignment
            common_dims = [dim for dim in transform.dims if dim in new_transform.dims and dim != 't']
            if len(common_dims) > 0:
                # Select t=0 if it exists in transform, then assign
                if 't' in transform.dims:
                    transform_slice = transform.sel(t=0)
                else:
                    transform_slice = transform
                new_transform.loc[{dim: transform_slice.coords[dim] for dim in common_dims}] = transform_slice
            transform = new_transform
        si_utils.set_sim_affine(
            target_sim,
            transform,
            transform_key=transform_key)


def get_sim_position_final(sim, position=None, transform_keys=None, get_center=False):
    # accepts either a sim or a msim (its scale0 sim is used) - only position/transform metadata
    # is ever read here, never pixel data
    if isinstance(sim, DataTree):
        sim = msi_utils.get_sim_from_msim(sim, scale='scale0')
    if position is None:
        position = si_utils.get_origin_from_sim(sim)
    if transform_keys is None or len(transform_keys) == 0:
        transform_keys = si_utils.get_tranform_keys_from_sim(sim)

    transforms = []
    transform_dims = []
    for transform_key in transform_keys:
        transform = si_utils.get_affine_from_sim(sim, transform_key)
        if 't' in transform.dims:
            transform = transform.isel(t=0)
        transforms.append(np.array(transform))
        transform_dims = transform['x_in'].data.tolist()
    transform = combine_transforms(transforms)

    new_position = apply_transform_dict([position], transform, transform_dims)[0]
    for dim in position.keys():
        if dim not in new_position:
            new_position[dim] = position[dim]
    if get_center:
        physical_size = get_sim_physical_size(sim)
        new_position = {dim: new_position[dim] + physical_size.get(dim, 0) / 2 for dim in new_position}
    return new_position


def group_sims_by_z(sims, positions=None):
    grouped_sims = []
    if positions is None:
        positions = [si_utils.get_origin_from_sim(sim) for sim in sims]
    z_positions = [position.get('z', 0) for position in positions]
    unique_z_values = len(set(z_positions)) == len(z_positions)
    if not unique_z_values:
        sims_by_z = {}
        for simi, z_pos in enumerate(z_positions):
            if z_pos is not None and z_pos not in sims_by_z:
                sims_by_z[z_pos] = []
            sims_by_z[z_pos].append(simi)
        grouped_sims = list(sims_by_z.values())
    if len(grouped_sims) == 0:
        grouped_sims = [list(range(len(sims)))]
    return grouped_sims


def calc_foreground_map(sims):
    # accepts either sims or msims (each msim's scale0 sim is used) - this genuinely needs
    # concrete pixel data (a median-image comparison across all sources), unlike the cheap
    # metadata-only helpers above, but the caller still shouldn't have to pre-extract a sims list
    sims = [msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
           for item in sims]
    if len(sims) <= 2:
        return [True] * len(sims)
    sims = [sim.squeeze().astype(np.float32) for sim in sims]
    median_image = calc_images_median(sims).astype(np.float32)
    difs = [np.mean(np.abs(sim - median_image), (0, 1)) for sim in sims]
    # or use stddev instead of mean?
    threshold = np.mean(difs, 0)
    #threshold, _ = cv.threshold(np.array(difs).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
    #threshold, foregrounds = filter_noise_images(channel_images)
    map = (difs > threshold)
    if np.all(map == False):
        return [True] * len(sims)
    return map


def normalise_sim(sim, transform_key, min, range, dtype):
    #image = (sim - min) / range
    image = ((sim - min) / range + 1) / 2   # extended range
    image = float2int_image(image.clip(0, 1), dtype)    # np.clip(image) is not dask-compatible, use image.clip() instead
    return si_utils.get_sim_from_array(
        image,
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=si_utils.get_origin_from_sim(sim),
        transform_key=transform_key,
        affine=si_utils.get_affine_from_sim(sim, transform_key),
        c_coords=sim.c.data,
        t_coords=sim.t.data
    )


def calc_normalise_stats(sims, use_global=True):
    """Per-source (mean, stddev) normalisation stats and dtype - accepts either sims or msims
    (each msim's scale0 sim is used). Split out from normalise_sims so a caller that only wants
    the statistics (e.g. to then apply them separately across a whole pyramid via map_msim_levels)
    doesn't have to build - and immediately discard - a full set of already-normalised sims.
    """
    sims = [msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
           for item in sims]
    dtype = sims[0].dtype
    # global mean and stddev
    if use_global:
        mins = [np.mean(sim, dtype=np.float32) for sim in sims]
        ranges = [np.std(sim, dtype=np.float32) for sim in sims]
        #min, max = get_image_window(sim, low=0.01, high=0.99)
        #range = max - min
        min = np.mean(mins)
        range = np.mean(ranges)
        stats = [(min, range)] * len(sims)
    else:
        stats = [(np.mean(sim, dtype=np.float32), np.std(sim, dtype=np.float32)) for sim in sims]
    return stats, dtype


def normalise_sims(sims, transform_key, use_global=True):
    stats, dtype = calc_normalise_stats(sims, use_global=use_global)
    return [normalise_sim(sim, transform_key, min, range, dtype) for sim, (min, range) in zip(sims, stats)]


def gaussian_filter_sim(sim, transform_key, sigma):
    image = np.asarray(sim)
    spatial_dims = set(si_utils.get_spatial_dims_from_sim(sim))
    sigma_by_axis = [sigma if dim in spatial_dims else 0 for dim in sim.dims]
    blurred_image = gaussian(image, sigma=sigma_by_axis, preserve_range=True)
    new_sim = si_utils.get_sim_from_array(
        blurred_image.astype(sim.dtype),
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=si_utils.get_origin_from_sim(sim),
        transform_key=transform_key,
        affine=si_utils.get_affine_from_sim(sim, transform_key),
        c_coords=sim.c.data,
        t_coords=sim.t.data
    )
    return new_sim


def get_sim_physical_size(sim):
    # accepts either a sim or a msim (its scale0 sim is used)
    if isinstance(sim, DataTree):
        sim = msi_utils.get_sim_from_msim(sim, scale='scale0')
    size = si_utils.get_shape_from_sim(sim)
    spacing = si_utils.get_spacing_from_sim(sim)
    physical_size = {dim: size[dim] * spacing.get(dim, 1) for dim in size}
    return physical_size


def calc_output_properties(sims, transform_key, output_spacing_method=None, z_scale=None):
    # accepts either sims or msims - each msim is converted to its scale0 sim right where needed
    # below (spacing/affine/origin metadata reads only, never pixel data - fusion.calc_fusion_
    # stack_properties itself only reads this same cheap per-sim metadata) instead of requiring
    # the caller to have already built a separate sims list just to call this function
    sims = [msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
           for item in sims]
    output_spacing = {}
    spacings = [si_utils.get_spacing_from_sim(sim) for sim in sims]
    dims = list(spacings[0])
    is_3d = (sims[0].sizes.get('z', 0) > 1)

    if output_spacing_method:
        output_spacing_method = output_spacing_method.lower()
    if not output_spacing_method or 'mean' in output_spacing_method:
        output_spacing = {dim: np.mean([spacing[dim] for spacing in spacings]) for dim in dims}
    elif 'max' in output_spacing_method:
        output_spacing = {dim: max([spacing[dim] for spacing in spacings]) for dim in dims}
    elif 'min' in output_spacing_method:
        output_spacing = {dim: min([spacing[dim] for spacing in spacings]) for dim in dims}

    if z_scale and 'z' in dims and not is_3d:
        output_spacing['z'] = z_scale
    output_properties = fusion.calc_fusion_stack_properties(
        sims,
        [si_utils.get_affine_from_sim(sim, transform_key) for sim in sims],
        output_spacing,
        mode='union',
    )
    if 'z' in output_properties['shape'] and not is_3d:
        z_positions = sorted(set([si_utils.get_origin_from_sim(sim).get('z', 0) for sim in sims]))
        z_shape = len(z_positions)
        if z_shape <= 1:
            z_shape = len(sims)
        output_properties['shape']['z'] = z_shape
    return output_properties


def get_properties_from_transform(transform):
    if 't' in transform.dims:
        transform = transform.sel(t=0)
    xtranslation = param_utils.translation_from_affine(transform)
    dims = xtranslation['x_in'].data.tolist()
    translation = {dim: xtranslation.sel(x_in=dim).item() for dim in dims}
    rotation = get_rotation_from_transform(transform)
    scale = get_scale_from_transform(transform)
    return translation, rotation, scale


def get_data_mapping(data, transform_key=None, transform=None, translation0=None, rotation=None):
    if rotation is None:
        rotation = 0

    if isinstance(data, DataTree):
        sim = msi_utils.get_sim_from_msim(data)
    else:
        sim = data
    translation = si_utils.get_origin_from_sim(sim)
    if 'z' not in translation and translation0 is not None and 'z' in translation0:
        translation['z'] = translation0['z']

    if transform is not None:
        translation1, rotation1, _ = get_properties_from_transform(transform)
        dims = set(list(translation) + list(translation1))
        translation = {dim: translation.get(dim, 0) + translation1.get(dim, 0) for dim in dims}
        rotation += rotation1

    if transform_key is not None:
        transform1 = sim.transforms.get(transform_key)
        if transform1 is not None:
            _, rotation1, _ = get_properties_from_transform(transform1)
            rotation += rotation1

    return translation, rotation


def extract_z_scale(positions, scales=None):
    z_scale = None

    if scales is not None:
        z_scale0 = np.mean([scale.get('z', 0) for scale in scales])
        if z_scale0 > 0:
            z_scale = z_scale0

    if z_scale is None:
        z_positions = [position.get('z') for position in positions if 'z' in position]
        if len(z_positions) > 1:
            diffs = np.diff(sorted(set(z_positions)))
            if len(diffs) > 0:
                z_scale = min(diffs)
    return z_scale


def _minimal_bb_vertices(points, return_edge_path=False):
    """Return the corners of an oriented bounding box around *points*.

    Corners are ordered around one face before the corresponding corners on
    the opposite face.  If ``return_edge_path`` is true, return the continuous
    path used to render all edges of a 3D box without diagonals.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must be an (N, 2) or (N, 3) array")
    if len(points) == 0:
        raise ValueError("at least one point is required")

    centered = points - points.mean(axis=0)
    if points.shape[1] == 2:
        hull = ConvexHull(points) if np.linalg.matrix_rank(centered) == 2 else None
        hull_points = points[hull.vertices] if hull is not None else points
        edges = np.diff(np.vstack((hull_points, hull_points[0])), axis=0)
        directions = edges[np.linalg.norm(edges, axis=1) > 0]
        if len(directions) == 0:
            axes = np.eye(2)
        else:
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            candidates = np.stack(
                (
                    directions,
                    np.column_stack((-directions[:, 1], directions[:, 0])),
                ),
                axis=1,
            )
            areas = []
            for candidate in candidates:
                projected = points @ candidate.T
                areas.append(np.prod(np.ptp(projected, axis=0)))
            axes = candidates[np.argmin(areas)]
    else:
        rank = np.linalg.matrix_rank(centered)
        candidates = []
        if rank == 3:
            hull = ConvexHull(points)
            for simplex, normal in zip(hull.simplices, hull.equations[:, :3]):
                z_axis = normal / np.linalg.norm(normal)
                face = points[simplex]
                for start, end in ((0, 1), (1, 2), (2, 0)):
                    x_axis = face[end] - face[start]
                    x_axis -= np.dot(x_axis, z_axis) * z_axis
                    length = np.linalg.norm(x_axis)
                    if length != 0:
                        x_axis /= length
                        y_axis = np.cross(z_axis, x_axis)
                        candidates.append(np.stack((x_axis, y_axis, z_axis)))

        if candidates:
            volumes = []
            for candidate in candidates:
                projected = points @ candidate.T
                volumes.append(np.prod(np.ptp(projected, axis=0)))
            axes = candidates[np.argmin(volumes)]
        else:
            # A deterministic orthonormal basis also gives sensible output for
            # lower-rank inputs, for which a 3D minimum-volume box is not unique.
            _, _, vh = np.linalg.svd(centered, full_matrices=True)
            axes = vh

    projected = points @ axes.T
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    face_order = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    corner_bits = face_order
    if points.shape[1] == 3:
        corner_bits = np.vstack(
            (
                np.column_stack((face_order, np.zeros(4, dtype=int))),
                np.column_stack((face_order, np.ones(4, dtype=int))),
            )
        )
    local_corners = mins + corner_bits * (maxs - mins)
    corners = local_corners @ axes

    if not return_edge_path:
        return corners
    if points.shape[1] != 3:
        raise ValueError("an edge path is only defined for 3D boxes")
    edge_path = [0, 1, 2, 3, 0, 4, 7, 3, 2, 6, 7, 4, 5, 6, 2, 1, 5]
    return corners[edge_path]


def set_oriented_bounding_box_edges(layer, shapes):
    """Correct napari-bbox's edge mesh for oriented 3D boxes.

    napari-bbox lexicographically sorts 3D vertices before applying a path
    intended for axis-aligned boxes.  Sorting destroys the topology of a
    rotated box and makes some of those edges diagonals.  Rebuild each private
    edge mesh from our known face-ordered vertices until napari-bbox preserves
    input ordering itself.
    """
    data_view = layer._data_view
    for index, shape in enumerate(shapes):
        bounding_box = data_view.bounding_boxes[index]
        if bounding_box.ndisplay != 3:
            raise RuntimeError(
                "The BoundingBoxLayer must be added to a 3D viewer before "
                "correcting its oriented edge mesh."
            )
        bounding_box._set_meshes(
            _minimal_bb_vertices(
                np.asarray(shape)[:, -3:],
                return_edge_path=True,
            ),
            closed=False,
            face=False,
        )
        data_view._update_mesh_vertices(index, edge=True)


def sims_from_sims_or_msims(items):
    # normalises a list of sims and/or msims to sims (each msim's scale0 sim is used)
    return [msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
           for item in items]


def stack_props_from_any(items, transform_key=None):
    """Normalises sims, msims or already-built stack-properties dicts to stack properties -
    the only thing the shapes path actually reads. A caller that has geometry but no image data
    (build_source_stack_props()) can therefore feed these functions directly, without an array
    having to be conjured up for it first.
    """
    stack_props = []
    for item in items:
        if isinstance(item, dict):
            stack_props.append(item)
            continue
        sim = msi_utils.get_sim_from_msim(item, scale='scale0') if isinstance(item, DataTree) else item
        if 't' in sim.dims:
            sim = sim.sel(t=0)
        stack_props.append(si_utils.get_stack_properties_from_sim(sim, transform_key=transform_key))
    return stack_props


def create_image_shapes(items, transform_key=None,  force_2d=False):
    # accepts sims, msims or stack properties - only position/size metadata is read, never pixel
    # data, and never anything that would make a source open its file
    all_stack_props = stack_props_from_any(items, transform_key)
    shapes = []
    is_multi_z_shapes = (len(set([props['origin'].get('z', 0) for props in all_stack_props])) > 1)
    for stack_props in all_stack_props:
        points = mv_graph.get_vertices_from_stack_props(stack_props)
        if points.shape[1] == 3 and (len(set(points[:, 0])) == 1 or force_2d):
            # remove constant z coordinate
            points = points[:, 1:]
        shape = _minimal_bb_vertices(points)
        if is_multi_z_shapes:
            z_position = stack_props['origin'].get('z', 0)
            shape = [[z_position] + list(element) for element in shape]
        shapes.append(shape)
    return shapes


def _filter_candidate_overlap_pairs(all_stack_props):
    """Every-pair candidates (np.triu_indices), narrowed to those whose axis-aligned bounding
    boxes actually overlap - a cheap, vectorized numpy broad phase in front of
    _get_overlap_bboxes' exact (linear-programming-based) intersection test, which is the
    expensive part: on an n-source project with no pair_registration graph yet to restrict
    candidates to (e.g. initial project load), the naive O(n^2) full pair set spends that solve
    on the vast majority of pairs that don't overlap at all, dominating redraw time once there
    are more than a few dozen sources. An AABB always contains the real (possibly rotated) box,
    so filtering on it can never drop a pair that genuinely overlaps.

    Also returns the per-sim mins/maxs themselves (not just which pairs survive) - the caller
    reuses them to draw an approximate overlap shape directly from each surviving pair's own
    AABB intersection, skipping the linprog solve entirely for this (pre-registration, no
    rotation yet) case.
    """
    mins = np.empty((len(all_stack_props), 3))
    maxs = np.empty((len(all_stack_props), 3))
    for index, stack_props in enumerate(all_stack_props):
        points = mv_graph.get_vertices_from_stack_props(stack_props)
        ndims = points.shape[1]
        mins[index] = 0
        maxs[index] = 0
        mins[index, :ndims] = points.min(axis=0)
        maxs[index, :ndims] = points.max(axis=0)
    overlaps = (
        np.all(mins[:, None, :] <= maxs[None, :, :], axis=-1)
        & np.all(mins[None, :, :] <= maxs[:, None, :], axis=-1)
    )
    iu = np.triu_indices(len(all_stack_props), 1)
    pairs = np.transpose(iu)[overlaps[iu]]
    return pairs, mins, maxs


def create_overlap_shapes(items, transform_key, pairs=None, force_2d=False, dtype=np.uint8):
    # accepts sims, msims or stack properties. Geometry is all this needs, and the common path
    # (broad phase + the AABB fast path below) works off stack properties alone - only
    # multiview_stitcher's exact overlap test insists on sims, so those are built, per pair that
    # actually reaches it, from the same properties (see get_pair_sim).
    all_stack_props = stack_props_from_any(items, transform_key)
    sims = None if any(isinstance(item, dict) for item in items) else sims_from_sims_or_msims(items)

    def get_pair_sim(index):
        if sims is not None:
            return squeeze_sim_transform_time(sims[index], transform_key)
        return sim_from_stack_props(all_stack_props[index], dtype, transform_key)

    shapes = []
    good_pairs = []
    is_multi_z_shapes = (len(set([props['origin'].get('z', 0) for props in all_stack_props])) > 1)
    # aabbs is only set for the broad-phase-discovered case below (no pair_registration graph
    # yet to restrict candidates to, e.g. initial project load) - there, sources still sit at
    # their raw source_metadata transform (no rotation applied yet), so each surviving pair's
    # own AABB intersection (mins/maxs, already computed by the broad phase) is exact, not just
    # a bound. Using it directly instead of _get_overlap_bboxes' exact (linprog-based)
    # intersection test skips that solver call - low-single-digit milliseconds even for a
    # trivial problem, and the dominant cost once there are thousands of candidate pairs -
    # entirely for this path. Once real pairs are given (post-registration, a much smaller,
    # already-curated set, and rotation may genuinely be present) the exact test is still used.
    aabbs = None
    if pairs is None:
        broad_phase_start = time.time()
        pairs, mins, maxs = _filter_candidate_overlap_pairs(all_stack_props)
        aabbs = (mins, maxs)
        logging.info(f'create_overlap_shapes: {len(pairs)} candidate pairs from'
                     f' {len(all_stack_props)} sims'
                     f' (broad phase: {time.time() - broad_phase_start:.1f}s)')
    n_exact_tests = 0
    exact_test_time = 0.0
    n_fast_shapes = 0
    for pair in pairs:
        props1 = all_stack_props[pair[0]]
        props2 = all_stack_props[pair[1]]
        shape_z_position = props1['origin'].get('z', 0)
        process_pair = True

        # Multi-section 2D data is promoted to singleton-z images for napari.
        # Calculate same-section overlaps in 2D because zero-thickness 3D
        # boxes produce invalid face normals in _get_overlap_bboxes.  The z
        # coordinate is restored to the resulting shape below.
        if (
            force_2d
            and props1['shape'].get('z') == 1
            and props2['shape'].get('z') == 1
        ):
            process_pair = (props1['origin'].get('z', 0) == props2['origin'].get('z', 0))

        if not process_pair:
            continue

        if aabbs is not None:
            mins, maxs = aabbs
            if force_2d:
                # sim1/sim2 still carry their (size-1) 'z' dim here - mins/maxs column 0 is that
                # z axis (matching the exact path's own points[:, 1:] below), columns 1/2 the
                # real spatial extent
                axes = slice(1, 3)
            elif 'z' in props1['shape'] and 'z' in props2['shape']:
                axes = slice(0, 3)
            else:
                axes = slice(0, 2)
            lo = np.maximum(mins[pair[0]], mins[pair[1]])[axes]
            hi = np.minimum(maxs[pair[0]], maxs[pair[1]])[axes]
            # broad_phase already guarantees lo <= hi in every axis (that's its own overlap
            # condition) - always a valid, non-empty box, no "no overlap" case to catch here
            corners = np.array(list(itertools.product(*zip(lo, hi))))
            shape = _minimal_bb_vertices(corners)
            n_fast_shapes += 1
            if is_multi_z_shapes:
                shape = [[shape_z_position] + list(element) for element in shape]
            shapes.append(shape)
            good_pairs.append(pair)
            continue

        # only from here on is an actual sim needed - the exact test below is multiview_stitcher's
        sim1, sim2 = get_pair_sim(pair[0]), get_pair_sim(pair[1])
        if force_2d:
            projected_sims = []
            for sim in (sim1, sim2):
                sim_2d = sim.squeeze('z', drop=True)
                sim_2d.attrs = dict(sim.attrs)
                sim_2d.attrs['transforms'] = dict(sim.attrs['transforms'])
                affine_2d = _adapt_transform_to_image_dims(
                    sim_2d,
                    sim_2d.attrs['transforms'][transform_key],
                    transform_key,
                )
                si_utils.set_sim_affine(sim_2d, affine_2d, transform_key)
                projected_sims.append(sim_2d)
            sim1, sim2 = projected_sims

        n_exact_tests += 1
        exact_test_start = time.time()
        try:
            # catch in case there is no overlap - _get_overlap_bboxes runs an exact
            # (scipy.optimize.linprog-based) intersection test per pair, a solver call that
            # costs low-single-digit milliseconds even for a trivial problem - the dominant
            # cost here once thousands of pairs reach it, see the logging below
            result = _get_overlap_bboxes(
                sim1,
                sim2,
                input_transform_key=transform_key,
                output_transform_key=transform_key,
            )
            points = result['intersection'].intersections
            if points.shape[1] == 3 and force_2d:
                # remove constant z coordinate
                points = points[:, 1:]
            shape = _minimal_bb_vertices(points)
            if is_multi_z_shapes:
                shape = [[shape_z_position] + list(element) for element in shape]
            shapes.append(shape)
            good_pairs.append(pair)
        except AttributeError:
            # ignore NoneType error if there is no overlap
            pass
        except ValueError as e:
            logging.exception(f'Error processing pair {pair}: {e}')
        finally:
            exact_test_time += time.time() - exact_test_start
    if n_exact_tests:
        logging.info(f'create_overlap_shapes: {n_exact_tests} exact intersection tests'
                     f' (of {len(pairs)} candidate pairs), {exact_test_time:.1f}s total'
                     f' ({1000 * exact_test_time / n_exact_tests:.1f}ms per test)')
    if n_fast_shapes:
        logging.info(f'create_overlap_shapes: {n_fast_shapes} shapes from broad-phase AABB '
                     f'intersection directly (no linprog)')
    return shapes, good_pairs


def get_overlap_images(sim1, sim2, transform_key):
    sims = [sim1.squeeze(), sim2.squeeze()]
    # functionality copied from registration.register_pair_of_msims()
    spatial_dims = si_utils.get_spatial_dims_from_sim(sim1)
    
    # Adapt transforms to match image dimensions (handles 3D transform on 2D images)
    for sim in sims:
        original_transform = sim.attrs['transforms'][transform_key]
        adapted_transform = _adapt_transform_to_image_dims(sim, original_transform, transform_key)
        sim.attrs['transforms'][transform_key] = adapted_transform
    
    result = _get_overlap_bboxes(
        sims[0],
        sims[1],
        input_transform_key=transform_key
    )
    lowers, uppers = result['lowers'], result['uppers']

    reg_sims_spacing = [
        si_utils.get_spacing_from_sim(sim) for sim in sims
    ]

    tol = 1e-6
    overlaps_sims = [
        si_utils.sim_sel_coords(
            sim, sel_dict={
                # add spacing to include bounding pixels
                dim: slice(
                    lowers[isim][idim] - tol - reg_sims_spacing[isim][dim],
                    uppers[isim][idim] + tol + reg_sims_spacing[isim][dim],
                )
                for idim, dim in enumerate(spatial_dims)
            }
        ) for isim, sim in enumerate(sims)
    ]

    sims_pixel_space = sims_to_intrinsic_coord_system(
        overlaps_sims[0],
        overlaps_sims[1],
        transform_key=transform_key,
        overlap_bboxes=(lowers, uppers),
    )

    fixed_data = sims_pixel_space[0].data
    moving_data = sims_pixel_space[1].data

    fixed_data = xr.DataArray(fixed_data, dims=spatial_dims)
    moving_data = xr.DataArray(moving_data, dims=spatial_dims)

    return fixed_data, moving_data, sims_pixel_space


def affine_from_intrinsic_affine(affine, sims_pixel_space, transform_key):
    affine_phys = get_affine_from_intrinsic_affine(
        data_affine=affine,
        sim_fixed=sims_pixel_space[0],
        sim_moving=sims_pixel_space[1],
        transform_key_fixed=transform_key,
        transform_key_moving=transform_key,
    )
    return affine_phys


def combine_transforms(transforms):
    combined_transform = None
    for transform in transforms:
        if combined_transform is None:
            combined_transform = transform
        else:
            combined_transform = np.dot(transform, combined_transform)
    return combined_transform


def squeeze_sim_dims(sim, transform_key):
    # very costly
    sim = sim.copy()
    if 't' in sim.dims:
        sim = sim.isel(t=0)
    if 'c' in sim.dims and sim.sizes.get('c', 0) <= 1:
        sim = sim.isel(c=0)
    affine = si_utils.get_affine_from_sim(sim, transform_key)
    if 't' in affine.dims:
        affine = affine.isel(t=0)
    if 'c' in affine.dims:
        affine = affine.isel(c=0)
    si_utils.set_sim_affine(sim, affine, transform_key)
    return sim


def squeeze_sim_transform_time(sim, transform_key):
    affine = si_utils.get_affine_from_sim(sim, transform_key)
    if 't' in affine.dims:
        affine = affine.isel(t=0)
        si_utils.set_sim_affine(sim, affine, transform_key)
    return sim


def make_sims_3d(sims, z_scale=None, positions=None):
    new_sims = []
    if not z_scale:
        z_scale = 1
    for index, sim in enumerate(sims):
        # check if already 3D
        if 'z' not in sim.dims:
            z_position = positions[index].get('z', index * z_scale)
            sim = sim.expand_dims({'z': [z_position]}, axis=-3)
        # set 3D affine transforms from 2D registration params
        for transform_key in si_utils.get_tranform_keys_from_sim(sim):
            transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
            if 4 not in transform.shape:
                transform_3d = param_utils.identity_transform(ndim=3)
                if 't' in transform.dims:
                    transform_3d.loc[{dim: transform.coords[dim] for dim in transform.sel(t=0).dims}] = transform.sel(t=0)
                else:
                    transform_3d.loc[{dim: transform.coords[dim] for dim in transform.dims}] = transform
                si_utils.set_sim_affine(sim, transform_3d, transform_key=transform_key)
        new_sims.append(sim)
    return new_sims


def make_sims_2d(sims):
    new_sims = []
    for index, sim in enumerate(sims):
        # check if already 2D
        if 'z' in sim.dims:
            sim = sim.squeeze('z')
        # set 2D affine transforms from 3D registration params
        for transform_key in si_utils.get_tranform_keys_from_sim(sim):
            transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
            if 3 not in transform.shape:
                has_t = 't' in transform.dims
                if has_t:
                    transform = transform.sel(t=0)
                transform = transform[:3, :3]
                if has_t:
                    transform.loc[{dim: transform.coords[dim] for dim in transform.sel(t=0).dims}] = transform.sel(t=0)
                si_utils.set_sim_affine(sim, transform, transform_key=transform_key)
        new_sims.append(sim)
    return new_sims


def promote_sim_to_3d(sim, z_position):
    # a sim/level with no native 'z' dim (e.g. a single 2D tile in a project where different
    # tiles sit at different z heights) gets a size-1 'z' dim added at its own z_position, and
    # every one of its transforms widened to 3D - the shared body behind make_msims_3d()'s
    # per-level promotion, also used directly by build_source_shape_sim() (which builds a plain
    # sim, never a msim, so map_msim_levels() doesn't apply)
    if 'z' not in sim.dims:
        sim = sim.expand_dims({'z': [z_position]}, axis=-3)
    for transform_key in si_utils.get_tranform_keys_from_sim(sim):
        transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        transform_3d = widen_xaffine_to_3d(transform)
        if transform_3d is not transform:
            si_utils.set_sim_affine(sim, transform_3d, transform_key=transform_key)
    return sim


def widen_xaffine_to_3d(transform):
    """A 2D affine widened into 3D, its own block embedded in a 3D identity - returned unchanged
    if it is already 3D. Shared by promote_sim_to_3d() and build_source_stack_props(), so a
    promoted sim and the stack properties derived without one carry the same transform.

    Placed by index into a plain numpy identity rather than by label into an xarray one: the
    label-based .loc assignment this replaces goes through xarray's alignment machinery, which
    cost 3.3ms per call - and make_msims_3d() makes one call per pyramid level of every source,
    so it was the larger half of promoting a 2D stack. The index mapping is derived from the
    same coordinate labels .loc matched on, so the result is identical (asserted by test).
    """
    if 4 in transform.shape:
        return transform
    if 't' in transform.dims:
        transform = transform.sel(t=0)
    labels_3d = ['z', 'y', 'x', '1']
    axes = [labels_3d.index(str(label)) for label in transform.coords['x_in'].values]
    widened = np.eye(len(labels_3d))
    widened[np.ix_(axes, axes)] = np.asarray(transform)
    return param_utils.affine_to_xaffine(widened)


def msim_is_already_3d(msim):
    """True when every level of `msim` already has a 'z' dim and a 3D transform - i.e. promoting
    it would produce exactly what is already there.

    A msim reaches make_msims_3d() more than once on the ordinary preview path: the viewer
    promotes each source's msim, takes a preview sub-pyramid from the result, and hands that to
    fuse(), which promotes again because the sources still sit at several z positions. The
    second pass rebuilds every level of every source to arrive back at the same geometry.
    """
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        dataset = msim[scale_key].ds
        if 'z' not in dataset['image'].dims:
            return False
        # each transform is its own data variable alongside 'image' (e.g. 'affine_metadata',
        # 3x3 while the sim is 2D, 4x4 once widened) - the same test widen_xaffine_to_3d applies
        transforms = [name for name in dataset.data_vars if name != 'image']
        if not transforms:
            return False
        if any(4 not in dataset[name].shape for name in transforms):
            return False
    return True


def make_msims_3d(msims, z_scale=None, positions=None):
    # msim-native equivalent of make_sims_3d: same promote_sim_to_3d() logic, applied
    # independently to every pyramid level via map_msim_levels - skipping any msim already in
    # that form, since rebuilding it would only reproduce it (see msim_is_already_3d)
    if not z_scale:
        z_scale = 1
    new_msims = []
    for index, msim in enumerate(msims):
        if msim_is_already_3d(msim):
            new_msims.append(msim)
            continue
        z_position = positions[index].get('z', index * z_scale) if positions else index * z_scale
        new_msims.append(map_msim_levels(
            msim, lambda sim, scale_key, z_position=z_position: promote_sim_to_3d(sim, z_position)))
    return new_msims


def make_msims_2d(msims):
    # msim-native equivalent of make_sims_2d
    new_msims = []
    for msim in msims:
        def level_func(sim, scale_key):
            if 'z' in sim.dims:
                sim = sim.squeeze('z')
            for transform_key in si_utils.get_tranform_keys_from_sim(sim):
                transform = si_utils.get_affine_from_sim(sim, transform_key=transform_key)
                if 3 not in transform.shape:
                    has_t = 't' in transform.dims
                    if has_t:
                        transform = transform.sel(t=0)
                    transform = transform[:3, :3]
                    if has_t:
                        transform.loc[{dim: transform.coords[dim] for dim in transform.sel(t=0).dims}] = transform.sel(t=0)
                    si_utils.set_sim_affine(sim, transform, transform_key=transform_key)
            return sim

        new_msims.append(map_msim_levels(msim, level_func))
    return new_msims


def extract_sims_from_msims(msims, sources, transform_key, target_scale):
    """Extract one working-resolution sim per source msim - the nearest native pyramid level to
    `target_scale`, resized only when no native level matches exactly. For plain scale0 extraction,
    use msi_utils.get_sim_from_msim(msim, scale='scale0') directly instead - this is only for the
    on-demand resize case (preprocess()'s `scale` override, a preview resolution smaller than any
    native level). msims are already properly chunked at creation (see ImageSource.get_msim) -
    only a genuinely resized array needs rechunking here.
    """
    sims = []
    for source, msim in zip(sources, msims):
        level, rescale, scale = get_level_from_scale(source, target_scale)
        sim = msi_utils.get_sim_from_msim(msim, scale=f'scale{level}')
        if any(value != 1 for value in rescale.values()):
            data = sim.data
            new_shape = [max(int(size / rescale.get(dim, 1)), 1) for dim, size in zip(sim.dims, data.shape)]
            data = resize(data, new_shape, preserve_range=True).astype(data.dtype)
            translation = si_utils.get_origin_from_sim(sim)
            affine = si_utils.get_affine_from_sim(sim, transform_key)
            channel_labels = list(sim.coords['c'].values) if 'c' in sim.coords else None
            # `scale` (from get_level_from_scale) only covers dims this source's own pixel size
            # has - a 2D source stacked into a 3D output_order (output has 'z', source doesn't)
            # would otherwise lose 'z' spacing entirely here, since get_sim_from_array requires
            # `scale` to be either None or cover every spatial dim already on the array
            full_scale = si_utils.get_spacing_from_sim(sim)
            if scale:
                full_scale.update(scale)
            sim = si_utils.get_sim_from_array(
                data, dims=list(sim.dims), scale=full_scale, translation=translation,
                affine=affine, transform_key=transform_key, c_coords=channel_labels)
            # resize() builds a brand new array - unlike a native level (already chunked at
            # msim-creation time), this one genuinely needs its own rechunk check
            sim = rechunk_if_monolithic(sim, default_chunk_size)
        sims.append(sim)
    return sims


def drop_finest_msim_level(msims):
    """Each msim minus its finest level, as a genuine (shallower) sub-pyramid - pure msim
    slicing, the same construction select_msim_subpyramid_at_scale() uses. Returns
    (msims, changed); an msim already down to a single level is returned untouched, and
    `changed` is False when none of them could be reduced any further.
    """
    result = []
    changed = False
    for msim in msims:
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        if len(scale_keys) > 1:
            changed = True
            result.append(DataTree.from_dict({f'scale{index}': msim[scale_key].ds
                                              for index, scale_key in enumerate(scale_keys[1:])}))
        else:
            result.append(msim)
    return result, changed


def estimate_fused_size(msims, transform_key, output_spacing_method=None, z_scale=None):
    """Bytes the fused output of `msims` would occupy, by the same reckoning MVSRegistration.
    fuse() reports as 'Fusing ...' - the output stack's own shape times the source dtype.

    Metadata only (calc_output_properties reads per-sim spacing/origin/affine, never pixel
    data), so it is cheap enough to ask before committing to a fusion: measured at 0.27s for
    328 sources, ~4s for 4733.
    """
    properties = calc_output_properties(msims, transform_key,
                                        output_spacing_method=output_spacing_method,
                                        z_scale=z_scale)
    itemsize = get_msim_image0(msims[0]).dtype.itemsize
    return int(np.prod([int(size) for size in properties['shape'].values()]) * itemsize), properties


def reduce_msims_to_fused_size(msims, transform_key, max_bytes=default_preview_max_bytes,
                               output_spacing_method=None, z_scale=None, max_steps=16,
                               label='preview'):
    """`msims` stepped to coarser pyramid levels until fusing them would produce at most
    max_bytes - the guard that keeps an on-screen preview from fusing an arbitrarily large
    stack.

    A preview occupies a few hundred pixels on screen whatever is behind it, so the size of the
    fused result is the thing worth bounding, rather than any particular way of choosing a
    level. preview_scale cannot do this job: it picks a level relative to each source's own
    pyramid, so the result still scales with the dataset, and the post-pre-processing preview
    is built from register_msims and never consults it at all. Working from the fused size
    instead covers both, and needs no per-source assumptions: it just drops the finest level
    while the result is too large and there is still a coarser one to drop to.

    Sources whose pyramid runs out first simply stop contributing reductions - hence the
    `changed` check, which ends the loop when nothing moved rather than spinning.
    """
    size, _ = estimate_fused_size(msims, transform_key, output_spacing_method, z_scale)
    if size <= max_bytes:
        return msims
    original_size = size
    reduced = msims
    for _ in range(max_steps):
        coarser, changed = drop_finest_msim_level(reduced)
        if not changed:
            logging.warning(
                f'{label}: fusing {print_hbytes(size)} exceeds the {print_hbytes(max_bytes)}'
                f' budget, but no source has a coarser pyramid level left to drop to'
                f' - converting these sources with a deeper pyramid would let this be reduced')
            return reduced
        reduced = coarser
        size, _ = estimate_fused_size(reduced, transform_key, output_spacing_method, z_scale)
        if size <= max_bytes:
            break
    logging.info(f'{label}: reduced to {print_hbytes(size)} '
                 f'(fusing at the requested resolution would have been {print_hbytes(original_size)},'
                 f' over the {print_hbytes(max_bytes)} budget)')
    return reduced

def select_msim_subpyramid_at_scale(msims, sources, target_scale, shortfall_warn_factor=4):
    """Select, per source, every native pyramid level from the nearest match to `target_scale`
    down to the coarsest, as a genuine (smaller) sub-pyramid msim - pure msim slicing, no sim
    extraction and no resize to an exact match.

    A source whose pyramid does not reach `target_scale` silently yields the finest level it
    does have, and everything downstream then fuses (and holds in memory) that much more than
    was asked for: the residual factor multiplies the output's linear size, so falling short by
    8x is 64x the pixels per plane to fuse. That is invisible in the result - it just looks
    slow - so log it once, with the shortfall, whenever it exceeds shortfall_warn_factor.

    The shortfall is measured only over dims the source could actually have downsampled. A
    size-1 dim (the 'z' every OME-Zarr tile carries, say) is identical at every level, so its
    residual is always the whole requested factor - counting it would report a 16x shortfall for
    a perfectly good pyramid whose x/y reach 8 of the 16 asked for.
    """
    result = []
    residuals = []
    for source, msim in zip(sources, msims):
        level, residual, _ = get_level_from_scale(source, target_scale)
        # only dims this source's own pyramid actually reduces: a dim whose coarsest level is
        # no smaller than its finest (a size-1 'z', or a z-stack whose levels only downsample
        # x/y) keeps the full target factor as its residual however complete the pyramid is, and
        # counting it made every OME-Zarr source - which always carries a 'z' - report the
        # maximum possible shortfall
        coarsest = source.scale_factors[-1] if source.scale_factors else {}
        reducible = [value for dim, value in residual.items() if coarsest.get(dim, 1) > 1]
        residuals.append(max(reducible) if reducible else 1)
        scale_keys = msi_utils.get_sorted_scale_keys(msim)[level:]
        result.append(DataTree.from_dict({f'scale{i}': msim[scale_key].ds
                                          for i, scale_key in enumerate(scale_keys)}))
    worst = max(residuals) if residuals else 1
    if worst >= shortfall_warn_factor:
        short = sum(1 for residual in residuals if residual >= shortfall_warn_factor)
        logging.warning(
            f'Preview scale {target_scale} not reachable for {short}/{len(residuals)} sources:'
            f' coarsest available level is up to {worst:.3g}x finer than requested, so the'
            f' preview fuses up to {worst ** 2:.3g}x more pixels per plane than intended.'
            f' Sources lacking coarse pyramid levels (e.g. a single-resolution OME-Zarr) are the'
            f' usual cause - re-converting them with a full pyramid restores the intended cost.')
    return result


def copy_transforms_to_msims(sources, target_msims, transform_key):
    """Copy `transform_key`'s affine from each `sources[i]` (a sim or msim) onto every scale of
    the corresponding `target_msims[i]` (msi_utils.set_affine_transform, one call per target - no
    per-level loop needed, the transform is the same at every scale). A msim source reads its
    transform directly (msi_utils.get_transform_from_msim) - no sim round-trip either.
    """
    dims = list(si_utils.get_origin_from_sim(get_msim_image0(target_msims[0])).keys())
    for source, target_msim in zip(sources, target_msims):
        if isinstance(source, DataTree):
            transform = msi_utils.get_transform_from_msim(source, transform_key)
        else:
            transform = si_utils.get_affine_from_sim(source, transform_key=transform_key)
        transform_dims = np.array(transform.coords['x_in'])
        if len(transform_dims) - 1 != len(dims):
            new_transform = param_utils.identity_transform(ndim=len(dims))
            common_dims = [dim for dim in transform.dims if dim in new_transform.dims and dim != 't']
            if len(common_dims) > 0:
                if 't' in transform.dims:
                    transform_slice = transform.sel(t=0)
                else:
                    transform_slice = transform
                new_transform.loc[{dim: transform_slice.coords[dim] for dim in common_dims}] = transform_slice
            transform = new_transform
        msi_utils.set_affine_transform(target_msim, transform, transform_key=transform_key)


def print_sim_info(data):
    # only convert msim -> sim when the input actually is a msim - a transform is readable
    # directly off a sim via si_utils.get_affine_from_sim, no need to build a msim just to
    # read one back via get_transform_from_msim
    if isinstance(data, DataTree):
        sim = msi_utils.get_sim_from_msim(data)
    else:
        sim = data

    print('dims', sim.dims)
    print('position dims', tuple(si_utils.get_origin_from_sim(sim).keys()))
    for transform_key in si_utils.get_tranform_keys_from_sim(sim):
        print(transform_key, si_utils.get_affine_from_sim(sim, transform_key).shape, end=' ')
    print()
