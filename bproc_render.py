# given a mesh and its angle, output two files:
# 1. 0.hdf5: save the whole points, normals, colors and depth
# 2. info.pkl: save the intrinsics, pose, angle, height, width
# the two output files is to be the input files of compute_point_cloud.py
# usage: blenderproc run ./bproc_render.py [--angle] [--width] [--height]
#        [--focal_x] [--focal_y] [--activate_antialiasing]  [--mesh_directory]
#        [--mesh_filename] [--output_directory]
import argparse

import blenderproc as bproc

from tools import generate_pose, save_info_dict, save_image, merge_path, generate_intrinsics

if __name__ == '__main__':
    bproc.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=str, default='random', help="which angle")
    parser.add_argument("--width", type=int, default=512, help="width of output image")
    parser.add_argument("--height", type=int, default=512, help="height of output image")
    parser.add_argument("--focal_x", type=int, default=400, help="x focal of output image")
    parser.add_argument("--focal_y", type=int, default=400, help="y focal of output image")
    parser.add_argument("--activate_antialiasing", type=bool, default=False, help="whether activate_antialiasing")
    parser.add_argument("--mesh_directory", type=str, default='./models/', help='mesh_directory')
    parser.add_argument("--mesh_filename", type=str, default='model_normalized.obj', help='mesh_filename directory')
    parser.add_argument("--output_directory", type=str, default='./bproc_render_output')
    parser.add_argument("--png_filename", type=str, default='rgb.png')
    parser.add_argument("--info_dict_filename", type=str, default='info_dict.pkl')
    args = parser.parse_args()

    angle = args.angle
    width = args.width
    height = args.height
    focal_x = args.focal_x
    focal_y = args.focal_y
    activate_antialiasing = args.activate_antialiasing
    mesh_directory = args.mesh_directory
    mesh_filename = args.mesh_filename
    output_directory = args.output_directory
    png_filename = args.png_filename
    info_dict_filename = args.info_dict_filename

    objs = bproc.loader.load_obj(merge_path(args.mesh_directory, args.mesh_filename))

    intrinsics = generate_intrinsics(focal_x, focal_y, width, height)
    bproc.camera.set_intrinsics_from_K_matrix(intrinsics, width, height)

    pose = generate_pose(angle)
    bproc.camera.add_camera_pose(pose)

    bproc.renderer.enable_depth_output(activate_antialiasing=activate_antialiasing)
    bproc.renderer.enable_normals_output()

    data = bproc.renderer.render()

    bproc.writer.write_hdf5(output_directory, data)
    save_image(output_directory, png_filename, data['colors'][0])
    save_info_dict(output_directory, info_dict_filename, intrinsics, pose, angle, width, height)
