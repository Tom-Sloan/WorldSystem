"""
Rerun blueprint manager for SLAM visualization.

Creates structured layouts for 3D point clouds, video streams, and statistics.
"""

import rerun.blueprint as rrb
import logging

logger = logging.getLogger(__name__)


def create_slam_visualization_blueprint() -> rrb.Blueprint:
    """
    Create comprehensive SLAM visualization blueprint.

    Layout:
    - Left: 3D point cloud with camera trajectory
    - Right Top: Camera video feed
    - Right Middle: Top-down 2D view
    - Right Bottom: Statistics panel

    Returns:
        Rerun Blueprint object
    """
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            # Left panel: Main 3D view (shows both real-time and optimized clouds)
            rrb.Spatial3DView(
                name="🗺️ SLAM Reconstruction",
                origin="/slam/world",
                contents=[
                    "/slam/world/accumulated_cloud",      # Real-time (raw)
                    "/slam/world/optimized_cloud",         # Optimized (filtered+downsampled)
                    "/slam/world/camera_trajectory/**",
                    "/slam/world/keyframe_poses/**"
                ]
            ),

            # Right panel: Video + top-down + stats
            rrb.Vertical(
                # Camera video feed
                rrb.Spatial2DView(
                    name="📹 Camera Feed",
                    origin="/slam/camera",
                    contents=["/slam/camera/video"]
                ),

                # Top-down view (shows both clouds)
                rrb.Spatial3DView(
                    name="📍 Top-Down View",
                    origin="/slam/world",
                    contents=[
                        "/slam/world/accumulated_cloud",
                        "/slam/world/optimized_cloud",
                        "/slam/world/camera_trajectory/**"
                    ]
                ),

                # Statistics
                rrb.TextDocumentView(
                    name="📊 Statistics",
                    origin="/slam/stats",
                    contents=["/slam/stats/**"]
                ),

                row_shares=[2, 2, 1]
            ),

            column_shares=[2, 1]
        ),
        auto_layout=False
    )

    logger.info("Created SLAM visualization blueprint")
    return blueprint


def create_point_cloud_only_blueprint() -> rrb.Blueprint:
    """
    Create simplified point cloud-only blueprint (no video).

    Returns:
        Rerun Blueprint object
    """
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            # Main 3D view (shows both real-time and optimized clouds)
            rrb.Spatial3DView(
                name="🗺️ SLAM Point Cloud",
                origin="/slam/world",
                contents=[
                    "/slam/world/accumulated_cloud",      # Real-time (raw)
                    "/slam/world/optimized_cloud",         # Optimized (filtered+downsampled)
                    "/slam/world/camera_trajectory/**"
                ]
            ),

            # Statistics
            rrb.TextDocumentView(
                name="📊 Statistics",
                origin="/slam/stats",
                contents=["/slam/stats/**"]
            ),

            row_shares=[4, 1]
        ),
        auto_layout=False
    )

    logger.info("Created point cloud-only blueprint")
    return blueprint
