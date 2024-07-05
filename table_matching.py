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
    # print('------------PREPROCESS POINT CLOUD FUNCTION-------------')
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_copy = copy.deepcopy(pcd)
    pcd_down = pcd_copy.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 1.5
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    # print('--------------END OF PREPROCESS------------------')
    return pcd_down, pcd_fpfh

def prepare_dataset(source_pcd, target_mesh, voxel_size):
    print('------------------PREPARE DATASET FUNCTION---------------------')
    print(":: Load two point clouds and disturb initial pose.")
    source = source_pcd
    # target_mesh = o3d.io.read_triangle_mesh(path_t)
    # target_mesh.compute_vertex_normals()
    target_points = np.asarray(target_mesh.vertices)
    # assign y values to 0, flatten target
    target_points[:, 2] = 0
    target_mesh.vertices = o3d.utility.Vector3dVector(target_points)
    if not source.has_points():
        print(f"Error: Unable to load source point cloud from cluster")
        return None, None, None, None, None, None
    if not target_mesh.has_triangles():
        print(f"Error: Unable to load target point cloud from mesh")
        return None, None, None, None, None, None

    # o3d.visualization.draw_geometries([source], 'source init')
    print('len target_mesh.points:', len(target_mesh.vertices))
    # o3d.visualization.draw_geometries([target_mesh], 'target mesh init')
    target = target_mesh.sample_points_uniformly(number_of_points=len(source.points))
    scale_factor = 1000
    scaled_points = np.asarray(target.points) * scale_factor
    target.points = o3d.utility.Vector3dVector(scaled_points)

    target_center = np.mean(np.asarray(target.points), axis=0)
    print('Target center: ', target_center)
    print('len source.points:', len(source.points))
    print('len target.points:', len(target.points))
    # flatten source
    source_points = np.asarray(source.points)
    source_points[:, 2] = 0
    source.points = o3d.utility.Vector3dVector(source_points)

    # translation to one center
    source_center = np.mean(np.asarray(source.points), axis=0)
    print('Source center: ', source_center)
    center = (target_center+source_center)/2
    translation = source_center - target_center
    # translation = source_center
    # translation_vector = center
    print(f'Source - {source_center}, target - {target_center} and translate center - {translation} ')
    # translation_vector = source_center
    translation_vector = translation
    print('Translation vector: ', translation_vector)
    # source_center1 = np.mean(np.asarray(source.points), axis=0)
    # target_center1 = np.mean(np.asarray(target.points), axis=0)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    print('target down number of points: ', len(target_down.points))
    print('source down number of points: ', len(source_down.points))
    # o3d.visualization.draw_geometries([target_down], "target")
    # o3d.visualization.draw_geometries([source_down], 'source')
    print('----------------------END OF PREPARING------------------------')
    return source, target, source_down, target_down, source_fpfh, target_fpfh, translation_vector

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
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.pi * 2 * 0.07),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold), o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(8000000, 600))
    print('------------------------END OF EXECUTION--------------------------')
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, tranformation_matrix, voxel_size):
    # print('------------------------REFINE REGISTRATION----------------------')
    distance_threshold = 20
    radius_normal = voxel_size * 1.5
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, tranformation_matrix,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print('------------------------END OF REFINE REGISTRATION-----------------------')
    return result

    #
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, transformation_matrix,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    # return result

def rotate_target(target_pcd, n):
    angle = np.pi/3
    # rotation = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    # rotation = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    # points = np.asarray(target_pcd)
    # target_pcd = target_pcd.rotate(rotation)
    rotation_matrix = [[np.cos(n*angle), -np.sin(n*angle), 0], [np.sin(n*angle), np.cos(n*angle), 0], [0, 0, 1]]
    return target_pcd, rotation_matrix

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

def transformation_matrix(rotation, translation):
    # assert rotation.shape == (3,3)
    # translation = 100*translation
    assert translation.shape == (3,) or translation.shape == (3,1)

    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation.reshape(3)
    return transformation

def compute_length_width(pcd):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    length = max_bound[0] - min_bound[0]
    width = max_bound[1] - min_bound[1]

    return length, width



def main():
    pcd = o3d.io.read_point_cloud('/home/user/Downloads/table.ply')
    # obj = o3d.io.read_point_cloud('/home/user/Downloads/75734416.obj')
    obj = o3d.io.read_triangle_mesh('/home/user/Downloads/75734416.obj')
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

    colors = np.random.rand(max_label + 1, 3)
    for i, cluster in enumerate(clusters):
        cluster.paint_uniform_color(colors[i])

    o3d.visualization.draw_geometries(clusters)

    # for i in range(max_label + 1):
    #     pcd0 = o3d.io.read_point_cloud(f'/home/user/PycharmProjects/images_python/matching_details{i}.ply')
    #     o3d.visualization.draw_geometries([pcd0])


    matched_pcd = []

    for cluster in clusters:

        voxel_size = 3
        source, target, source_down, target_down, source_fpfh, target_fpfh, translation = prepare_dataset(cluster,
                                                                                                          obj,
                                                                                                          voxel_size)
        if abs(len(source_down.points) - len(target_down.points)) > 40:
            print('__________NO MATCH_________')
            source.paint_uniform_color([1, 0, 0])
            matched_pcd.append(source)
            continue
        if source is None or target is None:
            print("Error: Failed to prepare the dataset.")
            return

        done = False
        final_transformation = np.eye(4)
        j = 0
        while not done:
            if j == 5:
                target = mirror_point_cloud(target)
                target_down = mirror_point_cloud(target_down)
                print('MIRRORED')
            target, rotation = rotate_target(target, j + 1)
            print("Rotation matrix:", rotation)
            # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            transformation = transformation_matrix(rotation, translation)
            print('Transformation matrix:\n ', transformation)
            # print("RANSAC result:", result_ransac, result_ransac.transformation)
            result_icp = refine_registration(target, source, target_fpfh, source_fpfh, transformation, voxel_size)
            print("ICP result:", result_icp)
            print("ICP transform:\n", result_icp.transformation)
            # draw_registration_result(source_down, target_down, transformation)
            # draw_registration_result(source_down, target_down, result_icp.transformation)
            j += 1
            length_s, width_s = compute_length_width(source_down)
            print('length source: ', length_s, width_s)
            length_t, width_t = compute_length_width(target_down)
            print('length target: ', length_t, width_t)
            if result_icp.fitness >= 0.99:
                if result_icp.inlier_rmse < 0.2:
                    final_transformation = result_icp.transformation
                    print('___________MATCH____________')
                    source.paint_uniform_color([0, 1, 0])
                    matched_pcd.append(source)
                    done = True
            if j > 10:
                print('NO MATCH')
                # source.paint_uniform_color([1, 0, 0])
                # matched_pcd.append(source)
                done = True
        draw_registration_result(source_down, target_down, final_transformation)

    o3d.visualization.draw_geometries(matched_pcd)



if __name__ == '__main__':
    main()

