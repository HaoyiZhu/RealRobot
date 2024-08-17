"""
RealSense Camera Interface.

Author: Haoyi Zhu.
"""

import cv2
import numpy as np
import pyrealsense2 as rs

from .base import CameraBase


class RealSenseRGBDCamera(CameraBase):
    """
    RealSense RGBD Camera Interface, e.g., D415, D435, etc.
    """

    def __init__(
        self,
        serial: str,
        shm_name_rgb: str = None,
        shm_name_depth: str = None,
        resolution=(1280, 720),
        frame_rate: int = 30,
        align: bool = True,
        depth_scale: float = 1000.0,
        **kwargs,
    ):
        assert not str.isalpha(serial[0]), "Not implemented."

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.serial = serial
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.align = align
        self.depth_scale = float(depth_scale)
        self._configure_pipeline()
        self.align_to = rs.stream.color
        self.aligner = rs.align(self.align_to) if align else None

        super().__init__(
            shm_name_rgb=f"{serial}_{shm_name_rgb}" if shm_name_rgb else None,
            shm_name_depth=f"{serial}_{shm_name_depth}" if shm_name_depth else None,
            frame_rate=frame_rate,
            **kwargs,
        )

    @property
    def name(self):
        return self.serial

    def _configure_pipeline(self):
        self.config.enable_device(self.serial)
        self.config.enable_stream(
            rs.stream.depth,
            self.resolution[0],
            self.resolution[1],
            rs.format.z16,
            self.frame_rate,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.resolution[0],
            self.resolution[1],
            rs.format.rgb8,
            self.frame_rate,
        )
        self.pipeline.start(self.config)

    def get_state(self, color=True, depth=True):
        frames = self.pipeline.wait_for_frames()
        if self.align:
            frames = self.aligner.process(frames)

        if color:
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
            if not depth:
                return color_image
        if depth:
            depth_frame = frames.get_depth_frame()
            depth_image = (
                np.asanyarray(depth_frame.get_data()).astype(float) / self.depth_scale
            )
            if not color:
                return depth_image
        return color_image, depth_image

    def get_color(self):
        return self.get_state(color=True, depth=False)

    def get_depth(self):
        return self.get_state(color=False, depth=True)

    def caliberate_intrinsic(self):
        profile = self.pipeline.get_active_profile()
        intrinsic = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        camera_matrix = np.array(
            [
                [intrinsic.fx, 0, intrinsic.ppx],
                [0, intrinsic.fy, intrinsic.ppy],
                [0, 0, 1],
            ]
        )
        dist_coeffs = np.array(intrinsic.coeffs)
        return camera_matrix, dist_coeffs

    def caliberate_extrinsic(
        self,
        camera_matrix,
        dist_coeffs,
        aruco_dict=cv2.aruco.DICT_4X4_50,
        squares_vertically=7,
        squares_horizontally=5,
        square_length=0.035,
        marker_length=0.026,
        visualize=False,
    ):
        color_image = self.get_color()
        undistorted_color_image = cv2.undistort(
            color_image, camera_matrix, dist_coeffs, None
        )
        gray = cv2.cvtColor(undistorted_color_image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        board = cv2.aruco.CharucoBoard(
            (squares_vertically, squares_horizontally),
            square_length,
            marker_length,
            aruco_dict,
        )
        params = cv2.aruco.DetectorParameters()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(gray, marker_corners, marker_ids)
            charuco_retval, charuco_corners, charuco_ids = (
                cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board
                )
            )
            assert charuco_retval, "Charuco corners not found."

        # Calibrate camera
        retval, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(
            np.array(charuco_corners),
            np.array(charuco_ids),
            board,
            camera_matrix,
            dist_coeffs,
            np.empty(1),
            np.empty(1),
        )
        if not retval:
            raise Exception("Failed to estimate pose of the Charuco board.")

        # Get the extrinsic matrix, i.e., the transformation matrix from the camera to the world frame
        rvec_matrix = cv2.Rodrigues(rvecs)[0]
        extrinsic_matrix = np.hstack((rvec_matrix, tvecs))
        extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))

        if visualize:

            from src.utils.pointcloud_utils import rgbd_to_pointcloud

            cv2.drawFrameAxes(
                color_image,
                camera_matrix,
                dist_coeffs,
                rvecs,
                tvecs,
                marker_length,
            )

            color, depth = self.get_state()
            points, colors = rgbd_to_pointcloud(
                color,
                depth,
                camera_matrix,
                extrinsic_matrix,
                min_depth=0.5,
                max_depth=1.0,
                depth_scale=1.0,
            )
            return dict(
                extrinsic_matrix=extrinsic_matrix,
                points=points,
                colors=colors,
                color_image=color_image,
            )

        return dict(extrinsic_matrix=extrinsic_matrix)

    def calibrate(
        self,
        *,
        camera_matrix=None,
        dist_coeffs=None,
        aruco_dict=cv2.aruco.DICT_4X4_50,
        squares_vertically=7,
        squares_horizontally=5,
        square_length=0.035,
        marker_length=0.026,
        force_intrinsic=False,
        visualize=False,
    ):
        """
        Calibrate the camera with both intrinsic and extrinsic parameters.
        The extrinsic parameters are obtained by detecting *ChArUco* markers.
        """
        # caliberate intrinsics
        if camera_matrix is None or dist_coeffs is None or force_intrinsic:
            camera_matrix, dist_coeffs = self.caliberate_intrinsic()

        # caliberate extrinsics
        extrinsic_dict = self.caliberate_extrinsic(
            camera_matrix,
            dist_coeffs,
            aruco_dict=aruco_dict,
            squares_vertically=squares_vertically,
            squares_horizontally=squares_horizontally,
            square_length=square_length,
            marker_length=marker_length,
            visualize=visualize,
        )
        return dict(
            serial=self.serial,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            **extrinsic_dict,
        )
