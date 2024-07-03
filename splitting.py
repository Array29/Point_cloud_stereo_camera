#!/usr/bin/env python
import os

import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('/home/user/Downloads/table.ply')
obj = o3d.io.read_point_cloud('/home/user/Downloads/75734416.obj')
o3d.visualization.draw_geometries([pcd])

pcd = pcd.voxel_down_sample(voxel_size=0.02)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)
labels = np.array(pcd.cluster_dbscan(eps=1.5, min_points=10, print_progress=True))

max_label = labels.max()
print(f'Point cloud has {max_label + 1} clusters')

clusters = []
path = '/home/user/PycharmProjects/images_python'

for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    cluster = pcd.select_by_index(cluster_indices)
    clusters.append(cluster)
    o3d.io.write_point_cloud(os.path.join(path, f'matching_details{i}.ply'), cluster)

colors = np.random.rand(max_label + 1,3)
for i, cluster in enumerate(clusters):
    cluster.paint_uniform_color(colors[i])

o3d.visualization.draw_geometries(clusters)

for i in range(max_label + 1):
    pcd0 = o3d.io.read_point_cloud(f'/home/user/PycharmProjects/images_python/matching_details{i}.ply')
    o3d.visualization.draw_geometries([pcd0])

# o3d.visualization.draw_geometries([pcd])