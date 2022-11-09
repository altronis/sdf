import numpy as np
import open3d as o3d


def get_pc_points_and_normals_from_files(points_path, normals_path):
    with open(points_path, 'rb') as f:
        pc_points = np.load(f)
    with open(normals_path, 'rb') as f:
        pc_normals = np.load(f)
    return pc_points, pc_normals


if __name__ == '__main__':
    points_path = './output/point_cloud.npy'
    normals_path = './output/normals.npy'
    pc_points, pc_normals = get_pc_points_and_normals_from_files(points_path, normals_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points)
    pcd.normals = o3d.utility.Vector3dVector(pc_normals)
    # pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # mesh_path = './models/model_normalized.obj'
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # o3d.visualization.draw_geometries([mesh], point_show_normal=True)
