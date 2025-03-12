from multiview_stitcher import vis_utils

if __name__ == '__main__':
    paths = ['D:\slides\EM04573_01t\tiles.ome.zarr']

    vis_utils.view_neuroglancer(zarr_paths=paths)
