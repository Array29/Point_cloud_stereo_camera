#!/usr/bin/env python

import os
import open3d as o3d
import numpy as np
import copy

path = '/home/user/PycharmProjects/images_python'

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(transformation)
    # source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



def preprocess_point_cloud(pcd, voxel_size):
    print('------------PREPROCESS POINT CLOUD FUNCTION-------------')
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_copy = copy.deepcopy(pcd)
    pcd_down = pcd_copy.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 1.5
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print('--------------END OF PREPROCESS------------------')
    return pcd_down, pcd_fpfh

def prepare_dataset(path_s, path_t, voxel_size):
    print('------------------PREPARE DATASET FUNCTION---------------------')
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(path_s)
    target_mesh = o3d.io.read_triangle_mesh(path_t)
    # target_mesh.compute_vertex_normals()
    target_points = np.asarray(target_mesh.vertices)
    # assign y values to 0, flatten target
    target_points[:, 2] = 0
    target_mesh.vertices = o3d.utility.Vector3dVector(target_points)
    if not source.has_points():
        print(f"Error: Unable to load source point cloud from {path_s}")
        return None, None, None, None, None, None
    if not target_mesh.has_triangles():
        print(f"Error: Unable to load target point cloud from {path_t}")
        return None, None, None, None, None, None

    o3d.visualization.draw_geometries([source], 'source init')
    print('len target_mesh.points:', len(target_mesh.vertices))
    o3d.visualization.draw_geometries([target_mesh], 'target mesh init')
    target = target_mesh.sample_points_uniformly(number_of_points=len(source.points))
    scale_factor = 1000
    scaled_points = np.asarray(target.points) * scale_factor
    target.points = o3d.utility.Vector3dVector(scaled_points)

    target_center = np.mean(np.asarray(target.points), axis=0)
    # flatten source
    source_points = np.asarray(source.points)
    source_points[:, 2] = 0
    source.points = o3d.utility.Vector3dVector(source_points)

    # translation to one center
    print('Target center: ', target_center)
    source_center = np.mean(np.asarray(source.points), axis=0)
    print('Source center: ', source_center)
    center = (target_center+source_center)/2
    print('Center: ', center)
    target.translate(center)
    source.translate(-center)
    source_center1 = np.mean(np.asarray(source.points), axis=0)
    print('Source center after translation: ', source_center1)
    target_center1 = np.mean(np.asarray(target.points), axis=0)
    print('Target center after translation: ', target_center1)
    print('len source.points:', len(source.points))
    print('len target.points:', len(target.points))
    # o3d.visualization.draw_geometries([target])
    # target = o3d.utility.Vector3dVector(target_n)
    # print('_______PRINT TARGET_____')
    # o3d.visualization.draw_geometries([target])

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    print('target down number of points: ', len(target_down.points))
    print('source down number of points: ', len(source_down.points))
    o3d.visualization.draw_geometries([target_down], "target")
    o3d.visualization.draw_geometries([source_down], 'source')
    print('----------------------END OF PREPARING------------------------')
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    print('----------------------EXECUTE GLOBAL REGISTRATION---------------------')
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                3),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal()
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999))
    print('------------------------END OF EXECUTION--------------------------')
    return result
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    print('------------------------REFINE REGISTRATION----------------------')
    distance_threshold = voxel_size * 0.2
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    print('------------------------END OF REFINE REGISTRATION-----------------------')
    return result


def move_top_surface_to_origin(pcd):
    points = np.asarray(pcd.points)
    z_coords = points[:, 2]
    max_z = np.max(z_coords)
    threshold = 0.01 * (max_z - np.min(z_coords))

    top_surface_indices = np.where(z_coords >= max_z - threshold)[0]
    top_surface_points = points[top_surface_indices]
    center_top_surface = np.mean(top_surface_points, axis=0)
    translation = -center_top_surface
    pcd.translate(translation)

    return pcd, center_top_surface

def rotate_target(target_pcd):
    angle = np.pi/60
    # rotation = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    rotation = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    points = np.asarray(target_pcd)
    target_pcd = target_pcd.rotate(rotation)
    return target_pcd

def check_point_cloud_coordinates(pcd):
    points = np.asarray(pcd.points)
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    mean_coords = points.mean(axis=0)
    print("Point Cloud Coordinates:")
    print("Min: ", min_coords)
    print("Max: ", max_coords)
    print("Mean: ", mean_coords)


def compute_length_width(pcd):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    length = max_bound[0] - min_bound[0]
    width = max_bound[1] - min_bound[1]

    return length, width


def mirror_point_cloud(pcd):
    # o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
    center = np.mean(np.asarray(pcd.points), axis=0)
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center
    pcd.transform(translation_to_origin)
    mirror_transform = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    pcd.transform(mirror_transform)
    translation_back = np.eye(4)
    translation_back[:3, 3] = center
    pcd.transform(translation_back)
    # o3d.visualization.draw_geometries([pcd], window_name="Mirrored Point Cloud")
    return pcd



def main():
    path_s = '/home/user/PycharmProjects/images_python'
    path_t = '/home/user/Downloads'

    for j in range(16):
        source_path = os.path.join(path_s, f'matching_details{j}.ply')
        target_path = os.path.join(path_t, '75734416.obj')

        if not os.path.exists(source_path):
            print(f"Source file does not exist: {source_path}")
            return
        if not os.path.exists(target_path):
            print(f"Target file does not exist: {target_path}")
            return

        voxel_size = 3
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_path, target_path,
                                                                                             voxel_size)
        check_point_cloud_coordinates(source)
        check_point_cloud_coordinates(target)
        length_s, width_s = compute_length_width(source_down)
        print('length source: ', length_s, width_s)
        length_t, width_t = compute_length_width(target_down)
        print('length target: ', length_t, width_t)

        if source is None or target is None:
            print("Error: Failed to prepare the dataset.")
            return

        # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        # print("RANSAC result:", result_ransac, result_ransac.transformation)
        # draw_registration_result(source_down, target_down, result_ransac.transformation)
        check_point_cloud_coordinates(source_down)
        check_point_cloud_coordinates(target_down)
        done = False
        final_transformation = np.eye(4)
        i = 0
        while not done:
            if i == 120:
                source = mirror_point_cloud(source)
                source_down = mirror_point_cloud(source_down)
                print('____________-___-___MIRRORED__________________')
            target = rotate_target(target)
            target_down = rotate_target(target_down)
            result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
            icp_tr = result_icp.transformation
            print("ICP result:", result_icp, result_icp.transformation)
            # draw_registration_result(source_down, target_down, icp_tr)
            check_point_cloud_coordinates(source_down)
            check_point_cloud_coordinates(target_down)
            length_s, width_s = compute_length_width(source_down)
            print('length source: ', length_s, width_s)
            length_t, width_t = compute_length_width(target_down)
            print('length target: ', length_t, width_t)
            i = i + 1
            if result_icp.fitness >= 0.99 and result_icp.inlier_rmse < 2.0:
                final_transformation = icp_tr
                print('___________MATCH____________')
                # break
                done = True
            if i > 240:
                print('NO MATCH')
                done = True
        draw_registration_result(source_down, target_down, final_transformation)

    # result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, result_ransac)
    # icp_tr = result_icp.transformation
    # print("ICP result:", result_icp, result_icp.transformation)
    # draw_registration_result(source_down, target_down, icp_tr)
    # check_point_cloud_coordinates(source_down)
    # check_point_cloud_coordinates(target_down)
    # length_s, width_s = compute_length_width(source_down)
    # print('length source: ', length_s, width_s)
    # length_t, width_t = compute_length_width(target_down)
    # print('length target: ', length_t, width_t)



if __name__ == '__main__':
    main()

