import mcubes
import numpy as np
import torch


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
