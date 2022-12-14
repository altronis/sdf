import numpy as np
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


# Normalize points to unit sphere
def normalize_pc(pts):
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    max_dist = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    return pts / max_dist


# Sample points on the point cloud, then perturb them
def sample_near_pc(pts, num_samples, var):
    num_pts = pts.shape[0]
    idx = np.random.choice(num_pts, num_samples)
    sample = pts[idx]

    std = var ** 0.5
    noise = np.random.normal(scale=std, size=sample.shape)
    return sample + noise


# Sample uniformly within a unit sphere
def sample_unit_sphere(num_samples):
    u = np.random.rand(num_samples, 1)
    pts = np.random.normal(size=(num_samples, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, np.newaxis]
    pts *= u ** (1. / 3)
    return pts


# Sample near point cloud and within the unit sphere to build training data
def sample_training_pts(pc_points, num_samples):
    var_near = 0.00025
    var_far = 0.0025

    # Sample around 95% of the points near the surface (as per DeepSDF paper)
    # Sample the rest uniformly within the unit sphere
    num_samples_near = int(num_samples * 0.475)
    num_samples_far = num_samples_near
    num_samples_uniform = num_samples - num_samples_near - num_samples_far

    sample_near = sample_near_pc(pc_points, num_samples_near, var_near)
    sample_far = sample_near_pc(pc_points, num_samples_far, var_far)
    sample_uniform = sample_unit_sphere(num_samples_uniform)
    sampled_pc = np.concatenate([sample_near, sample_far, sample_uniform], axis=0)

    return sampled_pc


# Get the sdf of each point in sample_pts using the closest point in the point cloud and
# its normal vector.
def get_sdf(pc_pts, pc_normals, sample_pts):
    # For each sampled point, find the closest pc point and its normal vector
    nbrs = NearestNeighbors(n_neighbors=1).fit(pc_pts)
    closest_idx = nbrs.kneighbors(sample_pts, return_distance=False).flatten()
    closest_pts = pc_pts[closest_idx]
    closest_normals = pc_normals[closest_idx]

    # Using that info, calculate the point-to-plane distance
    dist = closest_normals * (sample_pts - closest_pts)
    sdf = np.sum(dist, axis=1)
    return sdf


# Given a path to a mesh file, get point cloud and its normals
def get_pc_points_and_normals_from_mesh(mesh):
    pc_points, faces = mesh.sample(count=50000, return_index=True)
    pc_normals = mesh.face_normals[faces]
    pc_points = normalize_pc(pc_points)
    return pc_points, pc_normals


# Given 2 paths to the point cloud and normal file, get point cloud and its normals
def get_pc_points_and_normals_from_files(points_path, normals_path):
    with open(points_path, 'rb') as f:
        pc_points = np.load(f)
    with open(normals_path, 'rb') as f:
        pc_normals = np.load(f)
    return pc_points, pc_normals


# Given path(s) to mesh or pc & normals data files, get the sampled 3D point coordinates and their SDF values.
def get_training_data(num_samples, points_path=None, normals_path=None, mesh_path=None, source='file'):
    if points_path is not None and normals_path is not None and source == 'file':
        pc_points, pc_normals = get_pc_points_and_normals_from_files(points_path, normals_path)
        sample_pts = sample_training_pts(pc_points, num_samples)
        sdf = get_sdf(pc_points, pc_normals, sample_pts)

    elif mesh_path is not None and source == 'mesh':
        mesh = trimesh.load(mesh_path, force='mesh')
        pc_points, pc_normals = get_pc_points_and_normals_from_mesh(mesh)
        sample_pts = sample_training_pts(pc_points, num_samples)
        sdf = get_sdf(pc_points, pc_normals, sample_pts)

    else:
        raise ValueError("Invalid source parameter, must be either 'file' or 'mesh' ")

    train_data = {
        'pts': sample_pts,
        'sdf': sdf
    }
    return train_data


def to_tensor(x):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(x).float().to(device)


class PointsDataset(Dataset):
    def __init__(self, num_samples, points_path=None, normals_path=None, mesh_path=None, source='file'):
        train_data = get_training_data(num_samples, points_path, normals_path, mesh_path, source)
        self.pts = to_tensor(train_data['pts'])
        self.sdf = to_tensor(train_data['sdf'])

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        return {'pts': self.pts[idx], 'sdf': self.sdf[idx]}
