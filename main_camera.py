#!/usr/bin/env python

import open3d as o3d
from Camera import Camera
def main():

    camera = Camera()
    color, depth = camera.start_stream()
    pcd = camera.capturePointCloud(color, depth)
    normal, picked_points = camera.normal(pcd)
    camera.split_point_cloud(pcd, normal, picked_points)
    # pcd1 = camera.capture_PC_depth(depth)
    # camera.dist_between_2points(pcd)
    # camera.dist_camera_point(pcd)
    camera.stop_stream(color)
    camera.stop_stream(depth)






if __name__ == '__main__':
    main()
