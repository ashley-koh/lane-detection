# Lane Detection Project Plan

## Project Overview

This project implements a lane detection system for a robotic car running on NVIDIA Jetson Xavier NX. The system:
- Captures camera images via ROS2
- Runs inference using a lightweight CNN model
- Outputs a lane offset value (-1 to 1) for steering control

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Lane Detection Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │   ROS2 Bag  │───▶│  Extraction  │───▶│ Preprocess  │───▶│  Annotate  │ │
│  │  Recording  │    │              │    │  & Crop     │    │ (Roboflow) │ │
│  └─────────────┘    └──────────────┘    └─────────────┘    └────────────┘ │
│                                                                    │       │
│  ┌─────────────────────────────────────────────────────────────────┼──────┐│
│  │                         Training Pipeline                       ▼      ││
│  │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────┐  ││
│  │  │   Convert    │───▶│    Train     │───▶│  Export (ONNX/TensorRT) │  ││
│  │  │ Annotations  │    │    Model     │    │                         │  ││
│  │  └──────────────┘    └──────────────┘    └─────────────────────────┘  ││
│  └────────────────────────────────────────────────────────┬──────────────┘│
│                                                           │               │
│  ┌────────────────────────────────────────────────────────▼──────────────┐│
│  │                         Deployment (ROS2)                             ││
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐  ││
│  │  │   Camera    │───▶│  Inference   │───▶│  Lane Offset Publisher  │  ││
│  │  │   Topic     │    │    Node      │    │   /lane_detection/offset│  ││
│  │  └─────────────┘    └──────────────┘    └─────────────────────────┘  ││
│  └───────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
lane-detection/
├── agents/
│   └── plan.md                    # This file
├── configs/
│   └── default.yaml               # Default configuration
├── src/
│   ├── dataset/                   # Dataset handling
│   │   ├── augmentations.py       # Albumentations transforms
│   │   └── dataset.py             # PyTorch Dataset class
│   ├── export/                    # Model export utilities
│   │   ├── to_onnx.py             # ONNX export
│   │   └── to_tensorrt.py         # TensorRT conversion
│   ├── extraction/                # Data extraction
│   │   └── extract_images.py      # ROS2 bag image extraction
│   ├── models/                    # Neural network architectures
│   │   ├── base.py                # Base model class
│   │   ├── efficientnet.py        # EfficientNet variants
│   │   ├── mobilenetv3.py         # MobileNetV3 (recommended)
│   │   ├── resnet.py              # ResNet variants
│   │   └── factory.py             # Model factory
│   ├── preprocessing/             # Data preprocessing
│   │   ├── convert_annotations.py # Roboflow COCO to CSV
│   │   ├── crop.py                # Image cropping
│   │   └── prepare_dataset.py     # Frame selection
│   ├── ros/                       # ROS2 integration
│   │   └── inference_node.py      # Inference ROS2 node
│   └── training/                  # Training utilities
│       ├── losses.py              # Loss functions
│       ├── metrics.py             # Evaluation metrics
│       └── trainer.py             # Training loop
├── tests/                         # Unit tests
├── main.py                        # CLI entry point
├── pyproject.toml                 # Project dependencies
└── README.md                      # User documentation
```

## Workflow

### Phase 1: Data Collection

1. **Record ROS2 bags** while driving the robot
   - Include diverse conditions (lighting, road textures)
   - Record multiple runs
   
2. **Extract frames** from recordings
   ```bash
   lane-detection extract --bag-path ./bags/run1 --topic /camera/image_raw
   ```

3. **Preprocess images** (crop to road region)
   ```bash
   lane-detection crop --input-dir data/raw --output-dir data/cropped --crop-region bottom_half
   ```

4. **Select diverse frames** for annotation
   ```bash
   lane-detection prepare --input-dir data/cropped --output-dir data/to_annotate --num-samples 400
   ```

### Phase 2: Annotation

1. Upload `data/to_annotate/` to **Roboflow**
2. Annotate lane center as a keypoint (single point)
3. Export as COCO Keypoints format
4. Convert to training format:
   ```bash
   lane-detection convert --roboflow-dir ./roboflow_export --output-dir data/annotated
   ```

### Phase 3: Training

1. **Train the model**
   ```bash
   lane-detection train \
     --data-dir data/annotated \
     --architecture mobilenetv3 \
     --epochs 100 \
     --batch-size 32
   ```

2. **Monitor training** with TensorBoard:
   ```bash
   tensorboard --logdir outputs/logs
   ```

3. **Evaluate** best checkpoint:
   - Check validation MAE (target: < 0.05)
   - Check "within 0.1" accuracy (target: > 90%)

### Phase 4: Deployment

1. **Export to ONNX**
   ```bash
   lane-detection export --checkpoint outputs/checkpoints/best.pt --output model.onnx
   ```

2. **Convert to TensorRT** (on Jetson)
   ```bash
   lane-detection export \
     --checkpoint outputs/checkpoints/best.pt \
     --output model.engine \
     --tensorrt --fp16
   ```

3. **Deploy ROS2 node**
   ```bash
   ros2 run lane_detection inference_node --ros-args \
     -p engine_path:=/path/to/model.engine \
     -p use_tensorrt:=true \
     -p crop_preset:=bottom_half
   ```

## Model Selection Guide

| Model | Parameters | Inference (Jetson) | Recommended For |
|-------|------------|-------------------|-----------------|
| MobileNetV3 Small | ~1.5M | ~5ms | Fastest, real-time |
| MobileNetV3 Large | ~4.2M | ~8ms | Better accuracy |
| EfficientNet-B0 | ~4M | ~12ms | Good balance |
| ResNet18 | ~11M | ~15ms | Higher accuracy |

**Recommendation:** Start with `mobilenetv3` (small) for real-time performance.

## Key Configuration Parameters

### Training
- `learning_rate: 0.001` - Base LR (AdamW)
- `weight_decay: 0.01` - L2 regularization
- `warmup_epochs: 5` - LR warmup
- `patience: 15` - Early stopping patience
- `backbone_lr_multiplier: 0.1` - Lower LR for pretrained backbone

### Augmentation
- `horizontal_flip_prob: 0.5` - Critical for lane balance
- `brightness/contrast: 0.2` - Lighting robustness
- `rotation: 10 degrees` - Perspective variation

### Export (Jetson)
- `fp16: true` - Enable for 2x speedup
- `int8: false` - Requires calibration, use if needed
- `max_batch_size: 1` - Single image inference

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Validation MAE | < 0.05 | Mean absolute error |
| Within 0.1 accuracy | > 90% | Predictions within 10% of true offset |
| Inference time (TensorRT FP16) | < 10ms | On Jetson Xavier NX |
| End-to-end latency | < 50ms | Camera to steering command |

## Troubleshooting

### Training Issues

**High validation loss:**
- Increase dropout (0.3 → 0.5)
- Add more augmentation
- Check annotation quality
- Try larger model

**Overfitting:**
- Increase weight_decay
- Add more training data
- Use freeze_backbone_epochs: 5
- Enable stronger augmentation

**Slow convergence:**
- Decrease learning_rate
- Increase warmup_epochs
- Check data normalization

### Deployment Issues

**TensorRT conversion fails:**
- Ensure CUDA/TensorRT versions match
- Reduce workspace_size if OOM
- Convert on target device (Jetson)

**High inference latency:**
- Enable FP16: `--fp16`
- Reduce input_size (224 → 192)
- Use smaller model (mobilenetv3)

**Poor real-time performance:**
- Check crop_preset matches training
- Verify image topic quality
- Profile with `ros2 topic hz`

## Future Improvements

1. **Multi-point lane detection** - Detect full lane boundaries
2. **Uncertainty estimation** - Confidence scoring
3. **Domain adaptation** - Train on simulation, deploy on real
4. **Temporal smoothing** - Filter predictions over time
5. **Multiple camera support** - Front + side cameras
