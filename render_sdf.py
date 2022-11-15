import mcubes
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torchvision
from pytorch3d.implicitron.models.renderer.base import EvaluationMode, ImplicitFunctionWrapper
from pytorch3d.implicitron.models.renderer.ray_tracing import RayTracing
from pytorch3d.implicitron.models.renderer.rgb_net import RayNormalColoringNetwork
from pytorch3d.implicitron.models.renderer.sdf_renderer import SignedDistanceFunctionRenderer
from pytorch3d.renderer import MultinomialRaysampler
from pytorch3d.renderer.cameras import look_at_view_transform, FoVPerspectiveCameras

from model import DeepSDF


def to_mesh(model, res):
    # Prepare 3D coordinates
    space = np.linspace(-1, 1, res)
    x, y, z = np.meshgrid(space, space, space)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    coords = np.concatenate((x, y, z), axis=1)

    model.eval()
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        coords = torch.from_numpy(coords).float().to(device)
        sdf = model(coords)

    sdf = sdf.detach().cpu().numpy()
    sdf = sdf.reshape((res, res, res))
    vertices, triangles = mcubes.marching_cubes(sdf, 0)
    return vertices, triangles


def visualize_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        sdf = self.model(kwargs['rays_points_world'])
        feats = torch.full((sdf.shape[0], 1), 0.5, device=sdf.device)
        return torch.cat([sdf, feats], dim=-1)


class SDFRenderer(nn.Module):
    def __init__(self, render_res):
        super().__init__()
        self.render_res = render_res

        dist = 2.5
        elev = 0.0
        azim = 0.0
        device = 'cuda:0'
        bg_color = torch.full((1,), -1, device=device)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        ray_tracer = RayTracing()

        self.sdf_renderer = SignedDistanceFunctionRenderer(bg_color=bg_color)
        rgb_network = RayNormalColoringNetwork(feature_vector_size=1, d_out=1)
        rgb_network.cuda()
        self.sdf_renderer._rgb_network = rgb_network
        self.sdf_renderer.ray_tracer = ray_tracer

        raysampler = MultinomialRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            image_width=render_res,
            image_height=render_res,
            n_pts_per_ray=128,
            min_depth=0.1,
            max_depth=2.5
        )
        self.ray_bundle = raysampler(cameras)
        self.object_mask = torch.ones(render_res ** 2, dtype=torch.bool, device=device)

    def forward(self, sdf_model):
        sdf_wrapper = ImplicitFunctionWrapper(ModelWrapper(sdf_model))
        render_output = self.sdf_renderer(ray_bundle=self.ray_bundle, implicit_functions=[sdf_wrapper],
                                          evaluation_mode=EvaluationMode.TRAINING, object_mask=self.object_mask)

        out_img = render_output.features.squeeze()[:, :, 0]
        out_img = torch.fliplr(torch.flipud(out_img))

        # Normalize
        obj_mask = out_img != -1
        obj_pixels = out_img[obj_mask]
        min_val = obj_pixels.min()
        max_val = obj_pixels.max()

        out_img = out_img - min_val
        out_img = out_img / (max_val - min_val)
        out_img = torch.where(obj_mask, out_img, 1.0)
        return out_img


def main():
    model = DeepSDF(use_dropout=False)
    model.load_state_dict(torch.load('model.pth'))
    model.cuda()
    model.eval()

    render_mode = 'sdf'  # 'sdf' or 'mesh'

    # Differentiable SDF rendering (Sphere Tracing)
    if render_mode == 'sdf':
        renderer = SDFRenderer(render_res=64)
        image = renderer(model)
        torchvision.utils.save_image(image, f'out.png')

    # Render as mesh (for debugging)
    else:
        vertices, triangles = to_mesh(model, 64)
        out_mesh_path = 'out.obj'
        mcubes.export_obj(vertices, triangles, out_mesh_path)
        visualize_mesh(out_mesh_path)


if __name__ == '__main__':
    main()