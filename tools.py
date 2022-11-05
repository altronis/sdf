import numpy as np
import pickle
from PIL import Image
import os
import torch
import torchvision
import h5py
import open3d as o3d
import cmudict
import trimesh

def generate_description(categories, angle=None):
    l = []
    a_or_an = lambda word: 'an' if cmudict.dict().get(word, word[0] in 'aeiou')[0][0][-1].isdigit() else 'a'

    for category in categories:
        name = f'{a_or_an(category)} {category}'
        l += [name, f'a photo of {name}']
        if angle is not None and angle != 'random':
            l.append(f'{a_or_an(angle)} {angle} perspective of {name}')
    return l

def merge_path(directory: str, filename: str):
    if directory[-1] != '/':
        return directory + '/' + filename
    return directory + filename

def generate_pose(angle):
    # View from back to front
    if isinstance(angle, str):
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
            raise ValueError(f'wrong angle input string: {angle}')
    elif isinstance(angle, list):
        if len(angle) == 3:
            angle_x, angle_y, angle_z = angle
        else:
            raise ValueError(f'wrong angle input list length: {len(angle)}')
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

def info_from_intrinsics(intrinsics):
    focal_x, focal_y, width, height = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2] * 2, intrinsics[1, 2] * 2
    return focal_x, focal_y, width, height

def save_image(output_directory: str, output_filename: str, image):
    os.system('mkdir -p ' + output_directory)
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        pil_image.save(merge_path(output_directory, output_filename))
    elif isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, merge_path(output_directory, output_filename))
    else:
        numpy_image = np.array(image)
        pil_image = Image.fromarray(numpy_image)
        pil_image.save(merge_path(output_directory, output_filename))

def load_hdf5(input_directory: str, hdf5_filename: str):
    with h5py.File(merge_path(input_directory, hdf5_filename), "r") as data:
        colors, depth, normals = np.array(data['colors'][:]), np.array(data['depth'][:]), np.array(data['normals'][:])
    return colors, depth, normals

def load_info_dict(input_directory: str, info_filename: str):
    with open(merge_path(input_directory, info_filename), 'rb') as f:
        d = pickle.load(f)
        intrinsics, pose, angle, width, height = d['intrinsics'], d['pose'], d['angle'], d['width'], d['height']
    return intrinsics, pose, angle, width, height

def save_info_dict(output_directory: str, output_filename: str, intrinsics, pose, angle, width, height):
    os.system('mkdir -p ' + output_directory)
    with open(merge_path(output_directory, output_filename), 'wb') as f:
        d = dict()
        d['intrinsics'], d['pose'], d['angle'], d['width'], d['height'] = intrinsics, pose, angle, width, height
        pickle.dump(d, f)

def save_numpy(output_directory: str, output_filename: str, array: np.ndarray):
    os.system('mkdir -p ' + output_directory)
    with open(merge_path(output_directory, output_filename), 'wb') as f:
        np.save(f, array)
    

def get_pc_points_and_normals(points_path=None, normals_path=None, mesh_path=None, source='file', normalize=True):
    if points_path is not None and normals_path is not None and source == 'file':
        with open(points_path, 'rb') as f:
            pc_points = np.load(f)
        with open(normals_path, 'rb') as f:
            pc_normals = np.load(f)
    elif mesh_path is not None and source == 'mesh':
        mesh = trimesh.load(mesh_path)
        pc_points = mesh.vertices
        pc_normals = mesh.vertex_normals
    else:
        raise ValueError('get_training_data function has wrong value')
    if normalize:
        pc_points -= np.mean(pc_points, axis=0)
        pc_points /= np.max(np.linalg.norm(pc_normals, axis=1)) * 1.03
    pc_normals = pc_normals / np.linalg.norm(pc_normals, axis=1, keepdims=True)
    return pc_points, pc_normals

def o3d_visualization(object_type, points_path=None, normals_path=None, mesh_path=None, normals_source='file', point_show_normal=True):
    if normals_path is not None:
        normals_source = 'file'
    if object_type == 'points':
        pcd = o3d.geometry.PointCloud()
        with open(points_path, 'rb') as f:
            pc_points = np.load(f)
        pcd.points = o3d.utility.Vector3dVector(pc_points)
        if normals_source == 'file':
            with open(normals_path, 'rb') as f:
                pc_normals = np.load(f)
            pcd.normals = o3d.utility.Vector3dVector(pc_normals)
        elif normals_source == 'estimate':
            pcd.estimate_normals()
        else:
            raise ValueError(f'normals source is invalid: {normals_source}')
        o3d.visualization.draw_geometries([pcd], point_show_normal=point_show_normal)
    elif object_type == 'mesh':
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        o3d.visualization.draw_geometries([mesh], point_show_normal=point_show_normal)
    else:
        raise ValueError(f'object_type is invalid: {object_type}')
