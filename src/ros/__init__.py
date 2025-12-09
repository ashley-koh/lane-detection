"""
ROS2 integration for lane detection.

This module provides ROS2 nodes for inference on the Jetson Xavier NX.
"""

# ROS2 imports are optional - only available on systems with ROS2
try:
    from .inference_node import LaneDetectionNode, main
except ImportError:
    # ROS2 not available
    pass

__all__ = ["LaneDetectionNode", "main"]
