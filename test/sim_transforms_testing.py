import numpy as np
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import param_utils, msi_utils


if __name__ == '__main__':
    ndim = 3 # works as expected
    #ndim = 2 # transform gets projected to 2D
    sim = si_utils.get_sim_from_array(np.zeros([10] * ndim))
    p = param_utils.identity_transform(3)
    si_utils.set_sim_affine(sim, p, 'registered')
    p_sim = si_utils.get_affine_from_sim(sim, 'registered')
    p_msim = msi_utils.get_transform_from_msim(msi_utils.get_msim_from_sim(sim), 'registered')
    print(p_sim.shape, p_msim.shape)
