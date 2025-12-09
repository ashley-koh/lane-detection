# Lane Detection

A lightweight lane detection system for robotic cars running on NVIDIA Jetson or Apple Silicon. The model outputs a lane offset value (-1 to 1) that can be used for steering control.

## Features

- Extract images from ROS2 bag files
- Preprocess and prepare images for annotation
- Convert Roboflow annotations to training format
- Train with multiple architectures (MobileNetV3, EfficientNet, ResNet)
- Export to ONNX and TensorRT for deployment
- ROS2 inference node for real-time operation

## Installation

Requires Python 3.10-3.13.

```bash
# Clone the repository
git clone <repo-url>
cd lane-detection

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Additional Dependencies

**For ROS2 integration** (on Ubuntu with ROS2 Humble):
```bash
sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-std-msgs ros-humble-cv-bridge
```

**For TensorRT export** (on Jetson):
TensorRT is pre-installed on Jetson. Ensure `tensorrt` and `pycuda` are available in your Python environment.

## Supported Platforms

| Platform | Device | Training | Inference |
|----------|--------|----------|-----------|
| NVIDIA Jetson | CUDA | Yes | Yes (TensorRT) |
| Apple Silicon | MPS | Yes | Yes (ONNX) |
| Linux/Windows (NVIDIA GPU) | CUDA | Yes | Yes |
| CPU | CPU | Yes (slow) | Yes |

Training automatically detects and uses the best available device (CUDA > MPS > CPU).

## Quick Start

### 1. Extract Images from ROS2 Bags

```bash
# List available topics
lane-detection extract --bag-path ./rosbags/run1 --list-topics

# Extract images
lane-detection extract --bag-path ./rosbags/run1 --topic /camera/image_raw --output-dir data/raw
```

### 2. Preprocess Images

```bash
# Crop to bottom half (road region)
lane-detection crop --input-dir data/raw --output-dir data/cropped --crop-region bottom-half

# List available crop presets
lane-detection crop --list-presets

# Select diverse frames for annotation
lane-detection prepare --input-dir data/cropped --output-dir data/to_annotate --num-samples 400
```

#### Crop Presets

| Preset | Description |
|--------|-------------|
| `bottom-half` | Bottom 50% of the image (full width) |
| `bottom-third` | Bottom 33% of the image (full width) |
| `bottom-left` | Bottom-left quarter of the image |
| `bottom-right` | Bottom-right quarter of the image |
| `center-strip` | Middle third horizontal strip |
| `lower-center` | Bottom 50% height, center 50% width |

Custom regions can also be specified as `x,y,width,height`.

### 3. Annotate with Roboflow

1. Upload `data/to_annotate/` to [Roboflow](https://roboflow.com)
2. Create a keypoint annotation project
3. Annotate the lane center as a single keypoint on each image
4. Export as **COCO Keypoints** format
5. Download and extract to `./roboflow_export/`

### 4. Convert Annotations

```bash
lane-detection convert --roboflow-dir ./roboflow_export --output-dir data/annotated
```

### 5. Train the Model

```bash
lane-detection train --data-dir data/annotated --architecture mobilenetv3 --epochs 100
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

### 6. Export for Deployment

```bash
# Export to ONNX
lane-detection export --checkpoint outputs/checkpoints/best.pt --output model.onnx

# Export to TensorRT (on Jetson)
lane-detection export --checkpoint outputs/checkpoints/best.pt --output model.engine --tensorrt --fp16
```

### 7. Test the Model

Test on a single image:
```bash
# Basic inference
lane-detection test --image test.jpg --checkpoint outputs/checkpoints/best.pt

# With visualization output
lane-detection test --image test.jpg --checkpoint best.pt --output result.jpg

# With crop preset (useful for raw camera images)
lane-detection test --image raw.jpg --checkpoint best.pt --crop bottom-half --output result.jpg
```

Batch test on a directory of images:
```bash
# Process all images in a directory
lane-detection test --image data/test_images/ --checkpoint best.pt --output results/

# Custom CSV output path
lane-detection test --image data/test_images/ --checkpoint best.pt --csv predictions.csv

# Generate a video from results
lane-detection test --image data/test_images/ --checkpoint best.pt --video results.mp4

# Video with custom frame rate (default: 10 fps)
lane-detection test --image data/test_images/ --checkpoint best.pt --video results.mp4 --fps 30

# With ONNX model
lane-detection test --image data/test_images/ --checkpoint model.onnx --output results/
```

Batch mode outputs:
- CSV file with predictions for each image (filename, path, offset)
- Optional visualizations saved to the output directory
- Optional video file combining all visualizations
- Statistics summary (mean, std, min, max, distribution)

### 8. Deploy with ROS2

```bash
ros2 run lane_detection inference_node --ros-args \
  -p engine_path:=/path/to/model.engine \
  -p use_tensorrt:=true \
  -p image_topic:=/camera/image_raw
```

The node publishes lane offset to `/lane_detection/offset` (std_msgs/Float32).

## CLI Commands

| Command | Description |
|---------|-------------|
| `extract` | Extract images from ROS2 bag files |
| `crop` | Crop and preprocess images |
| `prepare` | Select diverse frames for annotation |
| `convert` | Convert Roboflow annotations to training format |
| `train` | Train the lane detection model |
| `test` | Test model on a single image or batch |
| `export` | Export model to ONNX or TensorRT |
| `info` | Show model architecture information |

Run `lane-detection <command> --help` for detailed options.

## Model Architectures

| Architecture | Parameters | Recommended For |
|--------------|------------|-----------------|
| `mobilenetv3` | ~1.5M | Real-time on Jetson (default) |
| `mobilenetv3_large` | ~4.2M | Higher accuracy, still efficient |
| `mobilenetv3_v2` | ~1.5M | Alternative head design |
| `efficientnet_lite0` | ~3M | Lightweight EfficientNet |
| `efficientnet_lite1` | ~4M | Balanced EfficientNet |
| `efficientnet_lite2` | ~5M | More accurate EfficientNet |
| `efficientnet_b0` | ~4M | Standard EfficientNet |
| `resnet18` | ~11M | Baseline |
| `resnet34` | ~21M | Maximum accuracy |

View all architectures:
```bash
lane-detection info
```

## Configuration

Default configuration is in `configs/default.yaml`. Override via CLI arguments:

```bash
lane-detection train \
  --data-dir data/annotated \
  --architecture mobilenetv3 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --dropout 0.3
```

## Project Structure

```
lane-detection/
├── configs/
│   └── default.yaml          # Default configuration
├── src/
│   ├── dataset/              # Dataset and augmentations
│   ├── export/               # ONNX and TensorRT export
│   ├── extraction/           # ROS2 bag image extraction
│   ├── models/               # Neural network architectures
│   ├── preprocessing/        # Image preprocessing
│   ├── ros/                  # ROS2 inference node
│   └── training/             # Training loop and metrics
├── main.py                   # CLI entry point
└── pyproject.toml            # Project dependencies
```

## How It Works

The model takes a camera image as input and outputs a single value between -1 and 1:
- **-1**: Lane center is at the left edge of the image
- **0**: Lane center is in the middle of the image
- **+1**: Lane center is at the right edge of the image

This offset value can be fed directly into a PID controller or similar for steering.

## License

MIT
