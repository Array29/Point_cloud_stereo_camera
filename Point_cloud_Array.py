#!/usr/bin/env python

import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2
import open3d as o3d

dist = '/home/user/Downloads/OpenNI-Linux-x64-2.3/Redist'

openni2.initialize(dist)

if openni2.is_initialized():
    print('OpenNI initialized')
else:
    print('OPenNI is not initialized')


dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

# depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
# color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))

depth_stream.start()
color_stream.start()

depth_intrinsics = depth_stream.get_video_mode()
print(depth_intrinsics)
color_intrinsics = color_stream.get_video_mode()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=depth_intrinsics.resolutionX, resolutionY=depth_intrinsics.resolutionY,
                       fps=30))
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)


depth_scale = depth_stream.get_max_pixel_value()

try:
    while True:
        depth_frame = depth_stream.read_frame()
        color_frame = color_stream.read_frame()

        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)
        print('depth data: ', depth_data[0])
        color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(color_frame.height, color_frame.width, 3)
        depth_image = o3d.geometry.Image(depth_data)

        cv2.imshow('depth image', depth_data)
        color_image = o3d.geometry.Image(color_data)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale, convert_rgb_to_intensity=False)
        # cv2.imshow('rgbd image', rgbd_image)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_intrinsics.resolutionX, height=depth_intrinsics.resolutionY, fx=depth_intrinsics.resolutionX, fy=depth_intrinsics.resolutionY, cx=depth_intrinsics.resolutionX/2, cy=depth_intrinsics.resolutionY/2)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

        o3d.visualization.draw_geometries([pcd])

finally:

    depth_stream.stop()
    color_stream.stop()

    dev.close()
    openni2.unload()


