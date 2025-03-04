from multiview_stitcher import ngff_utils


def save_ome_ngff(filename, sim, channels=None, translation=None, rotation=None,
                  compression=None, pyramid_downsample=2):
    pyramid_downsample_dict = {}
    for dim in sim.dims:
        if dim in 'xy':
            pyramid_downsample_dict[dim] = pyramid_downsample
        else:
            pyramid_downsample_dict[dim] = 1
    ngff_utils.write_sim_to_ome_zarr(sim, filename,
                                     downscale_factors_per_spatial_dim=pyramid_downsample_dict, overwrite=True)
