"""
ROS2 inference node for lane detection.

This node subscribes to camera images, runs inference using the trained model,
and publishes the lane offset value for the robot controller.

Usage (after building):
    ros2 run lane_detection inference_node --ros-args \
        -p engine_path:=/path/to/model.engine \
        -p use_tensorrt:=true

Topics:
    Subscribed:
        /camera/image_raw (sensor_msgs/Image): Input camera images

    Published:
        /lane_detection/offset (std_msgs/Float32): Lane offset [-1, 1]
        /lane_detection/debug_image (sensor_msgs/Image): Debug visualization (optional)

Parameters:
    engine_path (str): Path to TensorRT engine or ONNX model
    use_tensorrt (bool): Whether to use TensorRT (default: True)
    input_size (int): Model input size (default: 224)
    publish_debug (bool): Publish debug visualization (default: False)
    crop_preset (str): Image crop preset (default: "bottom_half")
"""

from typing import Optional
import numpy as np

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32
    from cv_bridge import CvBridge

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 not available. This module requires ROS2 Humble or later.")

# OpenCV
try:
    import cv2
except ImportError:
    cv2 = None


class LaneDetectionNode(Node):
    """
    ROS2 node for lane detection inference.

    This node processes camera images and outputs a lane offset value
    that can be used by a robot controller for lane following.
    """

    def __init__(self):
        super().__init__("lane_detection_node")

        # Declare parameters
        self.declare_parameter("engine_path", "")
        self.declare_parameter("use_tensorrt", True)
        self.declare_parameter("input_size", 224)
        self.declare_parameter("publish_debug", False)
        self.declare_parameter("crop_preset", "bottom_half")
        self.declare_parameter("image_topic", "/camera/image_raw")

        # Get parameters
        self.engine_path = self.get_parameter("engine_path").value
        self.use_tensorrt = self.get_parameter("use_tensorrt").value
        self.input_size = self.get_parameter("input_size").value
        self.publish_debug = self.get_parameter("publish_debug").value
        self.crop_preset = self.get_parameter("crop_preset").value
        self.image_topic = self.get_parameter("image_topic").value

        if not self.engine_path:
            self.get_logger().error("No engine_path specified!")
            raise ValueError("engine_path parameter is required")

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load model
        self._load_model()

        # Setup crop parameters
        self._setup_crop()

        # QoS profile for camera (best effort for real-time)
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            camera_qos,
        )

        # Publishers
        self.offset_pub = self.create_publisher(Float32, "/lane_detection/offset", 10)

        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                Image, "/lane_detection/debug_image", 10
            )

        self.get_logger().info(f"Lane detection node initialized")
        self.get_logger().info(f"  Model: {self.engine_path}")
        self.get_logger().info(f"  TensorRT: {self.use_tensorrt}")
        self.get_logger().info(f"  Input size: {self.input_size}")
        self.get_logger().info(f"  Subscribed to: {self.image_topic}")

    def _load_model(self):
        """Load the inference model."""
        if self.use_tensorrt:
            try:
                from src.export.to_tensorrt import TensorRTInference

                self.model = TensorRTInference(self.engine_path)
                self.get_logger().info("Loaded TensorRT engine")
            except ImportError:
                self.get_logger().error(
                    "TensorRT not available. Install tensorrt and pycuda."
                )
                raise
        else:
            # Use ONNX Runtime
            try:
                import onnxruntime as ort

                # Use CUDA if available
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.model = ort.InferenceSession(self.engine_path, providers=providers)
                self.input_name = self.model.get_inputs()[0].name
                self.get_logger().info("Loaded ONNX model with ONNX Runtime")
            except ImportError:
                self.get_logger().error("ONNX Runtime not available")
                raise

    def _setup_crop(self):
        """Setup image cropping based on preset."""
        # Crop presets (will be applied to each image)
        self.crop_presets = {
            "none": lambda h, w: (0, h, 0, w),
            "bottom_half": lambda h, w: (h // 2, h, 0, w),
            "bottom_third": lambda h, w: (2 * h // 3, h, 0, w),
            "bottom_two_thirds": lambda h, w: (h // 3, h, 0, w),
            "center": lambda h, w: (h // 4, 3 * h // 4, w // 4, 3 * w // 4),
        }

        if self.crop_preset not in self.crop_presets:
            self.get_logger().warn(
                f"Unknown crop preset: {self.crop_preset}, using 'none'"
            )
            self.crop_preset = "none"

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input BGR image from camera

        Returns:
            Preprocessed image as (1, 3, H, W) float32 array
        """
        h, w = image.shape[:2]

        # Apply crop
        y1, y2, x1, x2 = self.crop_presets[self.crop_preset](h, w)
        cropped = image[y1:y2, x1:x2]

        # Resize to model input size
        resized = cv2.resize(
            cropped, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # Convert to NCHW format
        tensor = normalized.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)

        return tensor.astype(np.float32)

    def infer(self, input_tensor: np.ndarray) -> float:
        """
        Run inference on preprocessed input.

        Args:
            input_tensor: Preprocessed input (1, 3, H, W)

        Returns:
            Lane offset value [-1, 1]
        """
        if self.use_tensorrt:
            output = self.model.infer(input_tensor)
        else:
            # ONNX Runtime
            output = self.model.run(None, {self.input_name: input_tensor})[0]

        # Get scalar value
        offset = float(output.flatten()[0])

        # Clamp to valid range
        offset = max(-1.0, min(1.0, offset))

        return offset

    def image_callback(self, msg: Image):
        """
        Process incoming camera image.

        Args:
            msg: ROS2 Image message
        """
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Preprocess
            input_tensor = self.preprocess(cv_image)

            # Run inference
            offset = self.infer(input_tensor)

            # Publish offset
            offset_msg = Float32()
            offset_msg.data = offset
            self.offset_pub.publish(offset_msg)

            # Publish debug visualization
            if self.publish_debug:
                self._publish_debug(cv_image, offset, msg.header)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def _publish_debug(self, image: np.ndarray, offset: float, header):
        """
        Create and publish debug visualization.

        Args:
            image: Original camera image
            offset: Predicted lane offset
            header: ROS message header
        """
        debug_image = image.copy()
        h, w = debug_image.shape[:2]

        # Draw center line
        center_x = w // 2
        cv2.line(debug_image, (center_x, 0), (center_x, h), (0, 255, 0), 2)

        # Draw predicted offset
        offset_x = int(center_x + offset * (w // 2))
        cv2.line(debug_image, (offset_x, 0), (offset_x, h), (0, 0, 255), 3)

        # Draw offset arrow
        arrow_y = h // 2
        cv2.arrowedLine(
            debug_image,
            (center_x, arrow_y),
            (offset_x, arrow_y),
            (255, 0, 0),
            3,
            tipLength=0.3,
        )

        # Add text
        text = f"Offset: {offset:.3f}"
        cv2.putText(
            debug_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Convert to ROS message and publish
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    """Main entry point for the ROS2 node."""
    if not ROS2_AVAILABLE:
        print("Error: ROS2 is not available. Please source your ROS2 installation.")
        print("  source /opt/ros/humble/setup.bash")
        return 1

    rclpy.init(args=args)

    try:
        node = LaneDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
