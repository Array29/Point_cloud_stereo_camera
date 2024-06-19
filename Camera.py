#!/usr/bin/env python

import os
import numpy as np
import open3d as o3d
from openni import openni2
from primesense import _openni2
import cv2
import matplotlib.pyplot as plt


class Camera:
    def __init__(self):
        self.path = '/home/user/Downloads/OpenNI-Linux-x64-2.3/Redist'
        openni2.initialize(self.path)
        if openni2.is_initialized():
            print('initialized')
        else:
            print('cannot initialize')

        self.device = openni2.Device.open_any()

    def start_stream(self):
        depth_stream = self.device.create_depth_stream()
        color_stream = self.device.create_color_stream()
        self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        depth_stream.start()
        color_stream.start()
        return color_stream, depth_stream

    def captureRGB(self, stream):

        try:
            color_frame = stream.read_frame()
            color_data = np.frombuffer(color_frame.get_buffer_as_uint8(),
                                       dtype=np.uint8).reshape(color_frame.height,
                                                               color_frame.width, 3)
            # rgb = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            fig, axs = plt.subplots(1, 1)
            axs.imshow(color_data)
            axs.set_title('RGB image')
            plt.show()
            color_image = o3d.geometry.Image(color_data)
            path = '/home/user/Desktop/images_python'
            cv2.imwrite(os.path.join(path, 'calibration.jpg'), cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB))
            # cv2.imwrite(path, cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB))
            # cv2.imshow('RGB image', rgb)
        except:
            print('cannot execute')


    def captureDepth(self, stream):
        depth_frame = stream.read_frame()
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(),
                                   dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)
        normalized_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        display = np.uint8(normalized_data)
        equalized_depth = cv2.equalizeHist(display)
        # cv2.imshow('depth image', equalized_depth)
        fig, axs = plt.subplots(1, 1)
        axs.imshow(equalized_depth, cmap="gray")
        axs.set_title('depth image')
        plt.show()
        o3d.geometry.Image(depth_data)


    def capturePointCloud(self, color_stream, depth_stream):
        depth_scale = 1000
        print(depth_scale)
        depth_intrinsics = depth_stream.get_video_mode()
        print("intrinsics:", depth_intrinsics)
        # self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        while True:
            depth_frame = depth_stream.read_frame()
            color_frame = color_stream.read_frame()
            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(),
                                       dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)
            color_data = np.frombuffer(color_frame.get_buffer_as_uint8(),
                                       dtype=np.uint8).reshape(color_frame.height, color_frame.width, 3)
            color_image = o3d.geometry.Image(color_data)

            depth_image = o3d.geometry.Image(depth_data)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,
                                                                            depth_image,
                                                                            depth_scale,
                                                                            convert_rgb_to_intensity=False)
            # intrinsics = o3d.camera.PinholeCameraIntrinsic(
            #     width=depth_intrinsics.resolutionX,
            #     height=depth_intrinsics.resolutionY,
            #     fx=depth_intrinsics.resolutionX,
            #     fy=depth_intrinsics.resolutionY,
            #     cx=depth_intrinsics.resolutionX / 2,
            #     cy=depth_intrinsics.resolutionY / 2
            # )
            intrinsics = o3d.camera.PinholeCameraIntrinsic(width=depth_intrinsics.resolutionX,
                                                           height=depth_intrinsics.resolutionY,
                                                           fx=590, fy=590, cx=397, cy=264)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            reflection_matrix = np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            mirrored_pcd = pcd.transform(reflection_matrix)

            o3d.visualization.draw_geometries([mirrored_pcd])

            path = '/home/user/Desktop/images_python'
            o3d.io.write_point_cloud(os.path.join(path, 'point_cloud.pcd'), mirrored_pcd)
            return mirrored_pcd

    def capture_PC_depth(self, depth_stream):
        depth_scale = 1000
        depth_frame = depth_stream.read_frame()
        depth_intrinsics = depth_stream.get_video_mode()
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(),
                                   dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)

        min_val = np.nanmin(depth_data)
        max_val = np.nanmax(depth_data)
        normalized_depth = (depth_data - min_val) / (max_val - min_val) * 255
        normalized_depth = normalized_depth.astype(np.uint8)
        color_data = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        color_image = o3d.geometry.Image(color_data)

        depth_image = o3d.geometry.Image(depth_data)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,
                                                                        depth_image,
                                                                        depth_scale,
                                                                        convert_rgb_to_intensity=False)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width=depth_intrinsics.resolutionX,
                                                       height=depth_intrinsics.resolutionY,
                                                       fx=590, fy=590, cx=397, cy=264)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pcd])


    def calibration_photos(self, stream):
        stream.read_frame()
        photos = []
        for i in range(20):
            color_frame = stream.read_frame()
            color_data = np.frombuffer(color_frame.get_buffer_as_uint8(),
                                       dtype=np.uint8).reshape(color_frame.height,
                                                               color_frame.width, 3)
            cv2.imshow('rgb photo', color_data)
            if cv2.waitKey(0):
                photos.append(color_data)
                # cv2.imshow('captured photo', color_data)
            if cv2.waitKey(1) == ord('q'):
                break
        self.stop_stream(stream)
        return photos

    def dist_between_2points(self, pcd):
        img = o3d.visualization.VisualizerWithEditing()
        img.create_window()
        img.add_geometry(pcd)
        img.run()
        img.destroy_window()

        pick_points = img.get_picked_points()
        point1 = np.asarray(pcd.points[pick_points[0]])
        point2 = np.asarray(pcd.points[pick_points[1]])

        dist = np.linalg.norm(point1 - point2)
        print(f'distance: {dist} meters')

    def dist_camera_point(self, pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # Press 'Shift' + left mouse button to pick a point
        vis.destroy_window()

        picked_points = vis.get_picked_points()

        if len(picked_points) == 0:
            print("No point selected.")
            return

        point = np.asarray(pcd.points)[picked_points[0]]
        origin = np.asarray([0.0, 0.0, 0.0])
        dist = np.linalg.norm(point - origin)
        print(f"Distance between camera and point: {dist} meters")


    def split_point_cloud(self, pcd:o3d.geometry.PointCloud, normal, picked_points):
        # pcd = o3d.io.read_point_cloud("/home/user/PycharmProjects/Astracam/point_cloud.pcd")
        # vis = o3d.visualization.VisualizerWithEditing()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # vis.run()
        # vis.destroy_window()
        # picked_points = vis.get_picked_points()

        if len(picked_points) >= 2:
            print('Picked points')
            point2 = np.asarray(pcd.points)[picked_points[0]] # [0] - left object
            point1 = np.asarray(pcd.points)[picked_points[1]] # [1] - right object
            print('Picked points: ', point1, point2)
            middle = (point1 + point2) / 2
            print('Middle: ', middle)
            midpointX = middle[0]
            print('midpointX: ', midpointX)

            left_ind = [i for i in range(len(pcd.points)) if pcd.points[i][0] < midpointX]
            right_ind = [i for i in range(len(pcd.points)) if pcd.points[i][0] >= midpointX]

            left_points_array = np.asarray(pcd.points)[left_ind]
            right_points_array = np.asarray(pcd.points)[right_ind]

            left_side = o3d.geometry.PointCloud()
            right_side = o3d.geometry.PointCloud()
            left_side.points = o3d.utility.Vector3dVector(left_points_array)
            right_side.points = o3d.utility.Vector3dVector(right_points_array)

            if pcd.has_colors():
                left_colors_array = np.asarray(pcd.colors)[left_ind]
                right_colors_array = np.asarray(pcd.colors)[right_ind]

                left_side.colors = o3d.utility.Vector3dVector(left_colors_array)
                right_side.colors = o3d.utility.Vector3dVector(right_colors_array)

            angle = np.radians(90)
            k = normal / np.linalg.norm(normal)

            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            I = np.eye(3)
            R_left = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            R_right = I + np.sin(-angle) * K + (1 - np.cos(-angle)) * np.dot(K, K)

            right_side.rotate(R_right, center=point1)
            left_side.rotate(R_left, center=point2)

            o3d.visualization.draw_geometries([left_side, right_side])
            o3d.visualization.draw_geometries([left_side])
            o3d.visualization.draw_geometries([right_side])
        else:
            print("Please pick at least two points.")

    def normal(self, pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        picked_points = vis.get_picked_points()

        if len(picked_points)==3:
            points = np.asarray(pcd.points)[picked_points]
            p1, p2, p3 = points
            v1 = p2 - p1
            v2 = p3 - p1
            norm = np.cross(v1, v2)
            norm = norm / np.linalg.norm(norm)
            d = -np.dot(norm, p1)
            plane_model = np.append(norm, d)
            normal = np.array(plane_model[:3])

            return normal, picked_points

    def stop_stream(self, stream):
        stream.stop()
        self.device.close()
        openni2.unload()
