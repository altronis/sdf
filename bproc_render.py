# given a mesh and its angle, output two files:
# 1. 0.hdf5: save the whole points, normals, colors and depth
# 2. info.pkl: save the intrinsics, pose, angle, height, width
# the two output files is to be the input files of compute_point_cloud.py
# usage: blenderproc run ./bproc_render.py [--angle] [--width] [--height] 
#        [--focal_x] [--focal_y] [--activate_antialiasing]  [--mesh_directory] 
#        [--mesh_filename] [--output_directory]
import blenderproc as bproc
import argparse

# from tools import generate_pose, save_info_dict, save_image, merge_path, generate_intrinsics
import numpy as np
import pickle
import os
from PIL import Image

def merge_path(directory: str, filename: str):
    if directory[-1] != '/':
        return directory + '/' + filename
    return directory + filename

def generate_pose(angle='random'):
    # View from back to front
    if angle == 'back':
        angle_x, angle_y, angle_z = np.pi / 2, 0, 0
    elif angle == 'right':
        angle_x, angle_y, angle_z = np.pi / 2, 0, 3 * np.pi / 2
    elif angle == 'front':
        angle_x, angle_y, angle_z = np.pi / 2, 0, np.pi
    elif angle == 'left':
        angle_x, angle_y, angle_z = np.pi / 2, 0, np.pi / 2
    elif angle == 'top':
        angle_x, angle_y, angle_z = 0, 0, np.pi
    elif angle == 'bottom':
        angle_x, angle_y, angle_z = 0, np.pi, 0
    elif len(angle) == 3:
        angle_x, angle_y, angle_z = int(angle[0]) * np.pi / 2, int(angle[1]) * np.pi / 2, int(angle[2]) * np.pi / 2
    elif angle == 'random':
        angle_x, angle_y, angle_z = np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi    
    else:
        raise ValueError(f'wrong angle input format: {angle}')
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)
    pose = np.concatenate([np.concatenate([R, t], 1), np.array([[0, 0, 0, 1]])], 0)
    return pose

def generate_intrinsics(focal_x, focal_y, width, height):
    return np.array([[focal_x, 0, width / 2], [0, focal_y, height / 2], [0, 0, 1]])

def save_image(output_directory: str, output_filename: str, image):
    os.system('mkdir -p ' + output_directory)
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        pil_image.save(merge_path(output_directory, output_filename))
    # elif isinstance(image, torch.Tensor):
    #     torchvision.utils.save_image(image, merge_path(output_directory, output_filename))
    else:
        numpy_image = np.array(image)
        pil_image = Image.fromarray(numpy_image)
        pil_image.save(merge_path(output_directory, output_filename))

def save_info_dict(output_directory: str, output_filename: str, intrinsics, pose, angle, width, height):
    os.system('mkdir -p ' + output_directory)
    with open(merge_path(output_directory, output_filename), 'wb') as f:
        d = dict()
        d['intrinsics'], d['pose'], d['angle'], d['width'], d['height'] = intrinsics, pose, angle, width, height
        pickle.dump(d, f)

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
