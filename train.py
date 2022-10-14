import matplotlib.pyplot as plt
import mcubes
import open3d as o3d
import torch
import trimesh
from torch.utils.data import DataLoader

import config
from data import SDFDataset
from model import DeepSDF, sdf_loss
from render import to_mesh


def visualize_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    mesh_path = 'bunny.obj'
    train_data = SDFDataset(mesh_path, config.num_samples)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    model = DeepSDF(use_dropout=config.use_dropout)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    running_loss = 0.0
    visualize_mesh(mesh_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(config.epochs):
        for i, data in enumerate(train_loader):
            pts = data['pts']
            gt_sdf = data['sdf']

            optim.zero_grad()
            pred_sdf = model(pts)
            pred_sdf = pred_sdf.flatten()

            loss = sdf_loss(pred_sdf, gt_sdf, delta=config.delta)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if i % config.log_interval == config.log_interval - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config.log_interval:.3f}')
                running_loss = 0.0

    vertices, triangles = to_mesh(model, config.res)
    out_mesh_path = 'out.obj'
    mcubes.export_obj(vertices, triangles, out_mesh_path)
    visualize_mesh(out_mesh_path)
