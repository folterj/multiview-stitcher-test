import numpy as np
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import param_utils, msi_utils


def test_set_affine():
    ndim = 3 # works as expected
    #ndim = 2 # transform gets projected to 2D
    sim = si_utils.get_sim_from_array(np.zeros([10] * ndim))
    p = param_utils.identity_transform(3)
    si_utils.set_sim_affine(sim, p, 'registered')
    p_sim = si_utils.get_affine_from_sim(sim, 'registered')
    p_msim = msi_utils.get_transform_from_msim(msi_utils.get_msim_from_sim(sim), 'registered')
    print(p_sim.shape, p_msim.shape)


def test_set_affine_combining1():
    sim = si_utils.get_sim_from_array(np.zeros([10] * 3))
    p1 = param_utils.identity_transform(3)
    si_utils.set_sim_affine(
        sim,
        p1,
        transform_key='p1')

    p2 = param_utils.affine_to_xaffine(
        param_utils.affine_from_translation([2, 3, 0]))

    si_utils.set_sim_affine(sim, p2, transform_key='p2', base_transform_key='p1')

    print('p1', si_utils.get_affine_from_sim(sim, 'p1'))
    print('p2', si_utils.get_affine_from_sim(sim, 'p2'))


def test_set_affine_combining2():
    p1 = param_utils.identity_transform(2)
    sim = si_utils.get_sim_from_array(np.zeros([10] * 3), affine=p1, transform_key='p1')

    p2 = param_utils.affine_to_xaffine(
        param_utils.affine_from_translation([2, 3, 0]))

    si_utils.set_sim_affine(sim, p2, transform_key='p2', base_transform_key='p1')

    print('p1', si_utils.get_affine_from_sim(sim, 'p1'))
    print('p2', si_utils.get_affine_from_sim(sim, 'p2'))


if __name__ == '__main__':
    #test_set_affine()
    test_set_affine_combining1()
    test_set_affine_combining2()
