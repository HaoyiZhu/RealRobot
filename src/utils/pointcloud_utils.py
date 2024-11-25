from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d not imported")


def rgbd_to_pointcloud(
    colors: str,
    depths: str,
    camera_matrix: np.ndarray,
    extrinsic: Optional[np.ndarray] = np.eye(4),
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    depth_scale: Optional[float] = 1000.0,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """
    Convert RGB-D images to a colorful point cloud.

    Args:
        colors (``numpy.ndarray``): RGB image.
        depths (``numpy.ndarray``): Depth image.
        camera_matrix (``numpy.ndarray``): Shape of (2, 3) camera matrix (i.e. left-upper intrinsic).
                [[fx, 0, cx],
                    [0, fy, cy]].
        extrinsic (``numpy.ndarray``): Shape of (4, 4) extrinsic matrix. Default: ``numpy.eye(4)``.
        min_depth (``float``): Min depth value in meters.
        max_depth (``float``): Max depth value in meters.
        depth_scale (``float``): Scale ratio converting depth image from millimeters to meters. Default: ``1000.``.

    Return:
        A tuple (pcd, colors, depth)
        - ``o3d.geometry.PointCloud``: Converted point cloud.
        - ``numpy.array``: (W, H, 3) RGB image.
        - ``numpy.array``: (W, H) depth image. Invalid area is filled with 0.
    """
    colors = colors.copy().astype(float)
    depths = depths.copy().astype(float)

    depths[np.isnan(depths)] = 0
    depths /= depth_scale  # from millimeters to meters
    depths[depths <= min_depth] = 0
    depths[depths >= max_depth] = 0

    assert colors.shape[:2] == depths.shape[:2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > min_depth) & (points_z < max_depth)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32).reshape(-1, 3)

    colors = np.copy(colors[mask]).astype(np.float32).reshape(-1, 3)

    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.linalg.inv(extrinsic) @ points[:, :, None]
    points = points / points[:, -1:, :]
    points = points[:, :3].reshape(-1, 3)

    return points, colors


def construct_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Construct an open3d point cloud from points and colors.

    Args:
        points (``numpy.ndarray``): Shape of (N, 3) coordinates.
        colors (``numpy.ndarray``): Shape of (N, 3) RGB values.

    Return:
        An open3d point cloud (``open3d.geometry.PointCloud``).
    """
    colors = colors.astype(float)
    if colors.max() > 1:
        colors = colors / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def vis_pcds(*pcds):
    o3d.visualization.draw_geometries([*pcds])
