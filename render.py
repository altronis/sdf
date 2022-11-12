import torch
import torch.nn as nn
import torchvision
import numpy as np

from model import DeepSDF
import sphere_tracing
from tools import info_from_intrinsics

def translation(sdf, t):
    def wrapper(p):
        d = sdf(p - t)
        return d

    return wrapper


def compute_rotation_matrix(axes, angles):
    nx, ny, nz = torch.unbind(axes, dim=-1)
    c, s = torch.cos(angles), torch.sin(angles)
    rotation_matrices = torch.stack([
        torch.stack([nx * nx * (1.0 - c) + 1. * c, ny * nx * (1.0 - c) - nz * s, nz * nx * (1.0 - c) + ny * s], dim=-1),
        torch.stack([nx * ny * (1.0 - c) + nz * s, ny * ny * (1.0 - c) + 1. * c, nz * ny * (1.0 - c) - nx * s], dim=-1),
        torch.stack([nx * nz * (1.0 - c) - ny * s, ny * nz * (1.0 - c) + nx * s, nz * nz * (1.0 - c) + 1. * c], dim=-1),
    ], dim=-2)
    return rotation_matrices


def render(model, intrinsics, distance, azimuth, elevation, device):
    dtype = torch.float32

    num_iterations = 500
    convergence_threshold = 1e-3

    # ---------------- Intrinsic matrix ---------------- #
    # fx = fy = 256
    # cx = cy = 128
    # camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)
    camera_matrix = torch.from_numpy(intrinsics).to(device).type(dtype)
    focal_x, focal_y, width, height = info_from_intrinsics(intrinsics)

    # ---------------- Camera position ---------------- #
    distance = 2.5
    azimuth = np.pi / 6
    elevation = np.pi / 8

    camera_position = torch.tensor([
        +np.cos(elevation) * np.sin(azimuth),
        -np.sin(elevation),
        -np.cos(elevation) * np.cos(azimuth)
    ], device=device, dtype=dtype) * distance

    # ---------------- Camera rotation ---------------- #
    target_position = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    # target_position = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
    up_direction = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    camera_z_axis = target_position - camera_position
    camera_x_axis = torch.cross(up_direction, camera_z_axis, dim=-1)
    camera_y_axis = torch.cross(camera_x_axis, camera_z_axis, dim=-1)
    camera_rotation = torch.stack((camera_x_axis, camera_y_axis, camera_z_axis), dim=-1)
    camera_rotation = nn.functional.normalize(camera_rotation, dim=-2)

    # ---------------- Directional light ---------------- #
    light_directions = torch.tensor([1.0, -0.5, 0.0], device=device, dtype=dtype)

    # ---------------- Ray marching ---------------- #
    y_positions = torch.arange(height, dtype=camera_matrix.dtype, device=device)
    x_positions = torch.arange(width, dtype=camera_matrix.dtype, device=device)
    y_positions, x_positions = torch.meshgrid(y_positions, x_positions, indexing='ij')
    z_positions = torch.ones_like(y_positions)
    ray_positions = torch.stack((x_positions, y_positions, z_positions), dim=-1)
    ray_positions = torch.einsum('mn,...n->...m', torch.inverse(camera_matrix),  ray_positions)
    ray_positions = torch.einsum('mn,...n->...m', camera_rotation, ray_positions) + camera_position
    ray_directions = nn.functional.normalize(ray_positions - camera_position, dim=-1)

    # ---------------- Rendering ---------------- #
    sdf = translation(model, torch.tensor([0.0, -1.0, 0.0], device=device))

    surface_positions, converged = sphere_tracing.sphere_tracing(
        signed_distance_function=sdf,
        ray_positions=ray_positions,
        ray_directions=ray_directions,
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
    )
    surface_positions = torch.where(converged, surface_positions, torch.zeros_like(surface_positions))

    surface_normals = sphere_tracing.compute_normal(
        signed_distance_function=sdf,
        surface_positions=surface_positions,
    )
    surface_normals = torch.where(converged, surface_normals, torch.zeros_like(surface_normals))

    image = sphere_tracing.phong_shading(
        surface_normals=surface_normals,
        view_directions=camera_position - surface_positions,
        light_directions=light_directions,
        light_ambient_color=1.0,
        light_diffuse_color=1.0,
        light_specular_color=1.0,
        material_ambient_color=0.2,
        material_diffuse_color=0.7,
        material_specular_color=0.1,
        material_emission_color=0.0,
        material_shininess=64.0,
    )

    image = torch.where(converged, image, torch.ones_like(image))
    return image.squeeze()


if __name__ == '__main__':
    model = DeepSDF(use_dropout=False)
    model.load_state_dict(torch.load('model.pth'))
    model.cuda()
    model.eval()
    image = render(model)
    torchvision.utils.save_image(image, f'out.png')
