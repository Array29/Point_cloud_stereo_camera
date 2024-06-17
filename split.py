#!/usr/bin/env python

import open3d as o3d
import numpy as np

# Load a point cloud from a file (replace with your file path)
point_cloud = o3d.io.read_point_cloud("/home/user/PycharmProjects/Astracam/point_cloud.pcd")

# Define a splitting plane along the x-axis (e.g., at x = 0.5)
bbox = point_cloud.get_axis_aligned_bounding_box()

# Define your own min and max bounds for splitting
split_value = 0.5  # Example value for splitting

# Get points below the split value on the x-axis
below_split = point_cloud.select_by_index(
    [i for i in range(len(point_cloud.points)) if point_cloud.points[i][0] < split_value])

# Get points above the split value on the x-axis
above_split = point_cloud.select_by_index(
    [i for i in range(len(point_cloud.points)) if point_cloud.points[i][0] >= split_value])

# Visualize the results
below_split.paint_uniform_color([1.0, 0, 0])  # Red for below split
above_split.paint_uniform_color([0, 1.0, 0])  # Green for above split

o3d.visualization.draw_geometries([below_split, above_split])
