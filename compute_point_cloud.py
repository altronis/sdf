# given hdf5 file (saving the whole points, normals, colors and depth) and
# info.pkl file (saving the intrinsics, pose, angle, height, width),
# to compute valid (depth < threshold) point cloud and its normals and/or colors.
# Output: point_cloud.npy, normals.npy, colors.npy
# usage: python ./compute_point_cloud.py [--input_directory] [--output_directory] 
#        [--depth_threshold] [--hdf5_filename] [--info_filename] 
#        [--output_colors] [--output_normals] 
import numpy as np
from kornia.geometry.camera.perspective import unproject_points
import argparse
import torch

from tools import load_info_dict, load_hdf5, save_numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, default='./bproc_render_output')
    parser.add_argument("--output_directory", type=str, default='./bproc_render_output')
    parser.add_argument("--depth_threshold", type=int, default=10000)
    parser.add_argument("--hdf5_filename", type=str, default='0.hdf5')
    parser.add_argument("--info_dict_filename", type=str, default='info_dict.pkl')
    parser.add_argument("--output_colors", type=bool, default=False)
    parser.add_argument("--output_normals", type=bool, default=True)
    parser.add_argument("--point_cloud_filename", type=str, default='point_cloud.npy')
    parser.add_argument("--normals_filename", type=str, default='normals.npy')
    parser.add_argument("--colors_filename", type=str, default='colors.npy')
    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    depth_threshold = args.depth_threshold
    hdf5_filename = args.hdf5_filename
    info_dict_filename = args.info_dict_filename
    output_colors = args.output_colors
    output_normals = args.output_normals
    point_cloud_filename = args.point_cloud_filename
    normals_filename = args.normals_filename
    colors_filename = args.colors_filename

    colors, depth, normals = load_hdf5(input_directory, hdf5_filename)
    intrinsics, pose, angle, width, height = load_info_dict(input_directory, info_dict_filename)

    uv = np.arange(0, height*width, dtype=int).reshape((height, width))
    uv = np.concatenate([(uv % width).reshape(-1, 1), (uv // width).reshape(-1, 1)], axis=1) # (height*width, 2)
    depth = depth.reshape(-1, 1) # (height, width(, 1)) -> (height*width, 1)
    valid = depth[:, 0] < depth_threshold # (height*width,) 
    uv = uv[valid] # (# of valid, 2)
    depth = depth[valid] # (# of valid, 1)
    colors = colors.reshape(-1, 3) # (height, width, 3) -> (height*width, 3)
    colors = colors[valid] #  (# of valid, 3)
    normals = normals.reshape(-1, 3) # (height, width, 3) -> (height*width, 3)
    normals = normals[valid] #  (# of valid, 3)
    normals = (normals - 0.5) * 2
    normals[:, 1] = -normals[:, 1]
    normals[:, 2] = -normals[:, 2]
    pc = unproject_points(torch.tensor(uv), torch.tensor(depth), torch.tensor(intrinsics)).numpy()

    save_numpy(output_directory, point_cloud_filename, pc)
    with open(output_directory + 'point_cloud.npy', 'wb') as f:
        np.save(f, pc)
    if output_normals:
        save_numpy(output_directory, normals_filename, normals)
    if output_colors:
        save_numpy(output_directory, colors_filename, colors)
