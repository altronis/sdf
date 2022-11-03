# given a mesh and its angle, output two files:
# 1. 0.hdf5: save the whole points, normals, colors and depth
# 2. info.pkl: save the intrinsics, pose, angle, height, width
# the two output files is to be the input files of compute_point_cloud.py
# usage: blenderproc run ./bproc_render.py [--angle] [--width] [--height] [--focal] 
#        [--activate_antialiasing]  [--mesh_directory] [--mesh_file_name] 
#        [--output_directory]
import blenderproc as bproc
import numpy as np
import argparse
import os
from PIL import Image
import pickle

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
    else:
        angle_x, angle_y, angle_z = np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi    

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

if __name__ == '__main__':
    bproc.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=str, default='random', help="which angle")
    parser.add_argument("--width", type=int, default=512, help="width of output image")
    parser.add_argument("--height", type=int, default=512, help="height of output image")
    parser.add_argument("--focal", type=int, default=400, help="focal of output image")
    parser.add_argument("--activate_antialiasing", type=bool, default=False, help="whether activate_antialiasing")
    parser.add_argument("--mesh_directory", type=str, default='./models/', help='mesh_directory')
    parser.add_argument("--mesh_file_name", type=str, default='model_normalized.obj', help='mesh_file_name directory')
    parser.add_argument("--output_directory", type=str, default='./')
    args = parser.parse_args()
    
    angle = args.angle
    width = args.width
    height = args.height
    focal = args.focal
    activate_antialiasing = args.activate_antialiasing
    mesh_path = args.mesh_directory + args.mesh_file_name
    output_directory = args.output_directory

    objs = bproc.loader.load_obj(mesh_path)

    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(intrinsics, width, height)

    pose = generate_pose(angle)
    bproc.camera.add_camera_pose(pose)

    bproc.renderer.enable_depth_output(activate_antialiasing=activate_antialiasing)
    bproc.renderer.enable_normals_output()
    
    data = bproc.renderer.render()

    os.system('mkdir -p ' + output_directory + 'output')
    bproc.writer.write_hdf5(output_directory + "output/", data)
    
    image = Image.fromarray(data['colors'][0])
    image.save(output_directory + 'output/rgb_'+args.angle+'.png')
    print('Save image: ' + output_directory + 'output/rgb_'+args.angle+'.png')
    with open(output_directory + 'output/info.pkl', 'wb') as f:
        d = dict()
        d['intrinsics'] = intrinsics
        d['pose'] = pose
        d['angle'] = angle
        d['height'] = height
        d['width'] = width
        pickle.dump(d, f)
        print('Save info dict: ' + output_directory + 'output/info.pkl')
     