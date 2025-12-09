"""
TensorRT export for lane detection models.

This module provides utilities to convert ONNX models to TensorRT engines
for optimized inference on NVIDIA GPUs (including Jetson Xavier NX).

Note: TensorRT must be installed on the target system. On Jetson, it comes
pre-installed with JetPack. On desktop, install via:
    pip install tensorrt

For best results, build the TensorRT engine on the target device.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def check_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    try:
        import tensorrt as trt

        return True
    except ImportError:
        return False


def export_to_tensorrt(
    onnx_path: str | Path,
    output_path: str | Path,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_size_gb: float = 1.0,
    calibration_data: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Path:
    """
    Convert an ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to the ONNX model
        output_path: Path for the output TensorRT engine
        fp16: Enable FP16 precision (recommended for Jetson)
        int8: Enable INT8 precision (requires calibration data)
        max_batch_size: Maximum batch size for the engine
        workspace_size_gb: GPU workspace size in GB
        calibration_data: Calibration data for INT8 quantization
        verbose: Print progress information

    Returns:
        Path to the TensorRT engine file

    Raises:
        ImportError: If TensorRT is not installed
        RuntimeError: If conversion fails
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT is not installed. Install it with:\n"
            "  pip install tensorrt\n"
            "Or on Jetson, ensure JetPack is properly installed."
        )

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Converting ONNX to TensorRT: {onnx_path}")
        print(f"  FP16: {fp16}")
        print(f"  INT8: {int8}")
        print(f"  Max batch size: {max_batch_size}")

    # Create logger
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)

    if verbose:
        print("Parsing ONNX model...")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            error_msgs = []
            for i in range(parser.num_errors):
                error_msgs.append(str(parser.get_error(i)))
            raise RuntimeError(f"Failed to parse ONNX model:\n" + "\n".join(error_msgs))

    if verbose:
        print(f"Network inputs: {network.num_inputs}")
        print(f"Network outputs: {network.num_outputs}")

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_size_gb * 1024 * 1024 * 1024)
    )

    # Set precision flags
    if fp16 and builder.platform_has_fast_fp16:
        if verbose:
            print("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16 and not builder.platform_has_fast_fp16:
        print("Warning: FP16 requested but not supported on this platform")

    if int8 and builder.platform_has_fast_int8:
        if verbose:
            print("Enabling INT8 mode")
        config.set_flag(trt.BuilderFlag.INT8)

        if calibration_data is not None:
            # Set up INT8 calibrator
            calibrator = Int8EntropyCalibrator(calibration_data)
            config.int8_calibrator = calibrator
        else:
            print("Warning: INT8 enabled but no calibration data provided")
    elif int8 and not builder.platform_has_fast_int8:
        print("Warning: INT8 requested but not supported on this platform")

    # Set optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()

    # Get input tensor info
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # Set min/opt/max shapes (batch, channels, height, width)
    min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
    opt_shape = (
        max(1, max_batch_size // 2),
        input_shape[1],
        input_shape[2],
        input_shape[3],
    )
    max_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])

    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if verbose:
        print("Building TensorRT engine (this may take a while)...")

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    if verbose:
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"TensorRT engine saved: {output_path}")
        print(f"Engine size: {file_size:.2f} MB")

    return output_path


class Int8EntropyCalibrator:
    """INT8 calibrator for TensorRT using entropy calibration."""

    def __init__(
        self,
        calibration_data: np.ndarray,
        cache_file: str = "calibration.cache",
    ):
        """
        Initialize the calibrator.

        Args:
            calibration_data: Numpy array of shape (N, C, H, W) with calibration images
            cache_file: Path to cache calibration data
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "INT8 calibration requires pycuda. Install with:\n  pip install pycuda"
            )

        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0

        # Allocate device memory
        self.device_input = cuda.mem_alloc(calibration_data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        import pycuda.driver as cuda

        if self.current_index >= len(self.calibration_data):
            return None

        # Copy batch to device
        batch = np.ascontiguousarray(self.calibration_data[self.current_index])
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1

        return [int(self.device_input)]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def load_tensorrt_engine(engine_path: str | Path, verbose: bool = True):
    """
    Load a TensorRT engine from file.

    Args:
        engine_path: Path to the TensorRT engine file
        verbose: Print info

    Returns:
        TensorRT engine object
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT is not installed")

    engine_path = Path(engine_path)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if verbose:
        print(f"Loaded TensorRT engine: {engine_path}")

    return engine


class TensorRTInference:
    """
    TensorRT inference wrapper for easy deployment.

    Example usage:
        engine = TensorRTInference("model.engine")
        output = engine.infer(preprocessed_image)
    """

    def __init__(self, engine_path: str | Path):
        """
        Initialize TensorRT inference.

        Args:
            engine_path: Path to the TensorRT engine file
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "TensorRT inference requires tensorrt and pycuda.\n"
                "Install with: pip install tensorrt pycuda"
            )

        self.engine = load_tensorrt_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # Get shapes
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate GPU buffers for inference."""
        import pycuda.driver as cuda

        # Calculate sizes (handle dynamic batch with size 1)
        input_shape = list(self.input_shape)
        output_shape = list(self.output_shape)

        # Replace -1 (dynamic) with 1
        input_shape = [s if s > 0 else 1 for s in input_shape]
        output_shape = [s if s > 0 else 1 for s in output_shape]

        input_size = int(np.prod(input_shape) * np.float32().itemsize)
        output_size = int(np.prod(output_shape) * np.float32().itemsize)

        # Allocate device memory
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)

        # Create stream
        self.stream = cuda.Stream()

        # Store shapes for later
        self._input_shape = input_shape
        self._output_shape = output_shape

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.

        Args:
            input_data: Preprocessed input image as numpy array (N, C, H, W)

        Returns:
            Model output as numpy array
        """
        import pycuda.driver as cuda

        # Ensure input is contiguous float32
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)

        # Set input shape for dynamic batch
        batch_size = input_data.shape[0]
        input_shape = (batch_size,) + tuple(self._input_shape[1:])
        output_shape = (batch_size,) + tuple(self._output_shape[1:])

        self.context.set_input_shape(self.input_name, input_shape)

        # Allocate output buffer
        output_data = np.empty(output_shape, dtype=np.float32)

        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output from device
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)

        # Synchronize
        self.stream.synchronize()

        return output_data

    def __del__(self):
        """Clean up resources."""
        # CUDA resources are cleaned up automatically by pycuda
        pass


def convert_onnx_to_tensorrt_cli():
    """Command-line interface for ONNX to TensorRT conversion."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to TensorRT engine"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output TensorRT engine path",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision (default: True)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 precision",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 precision (requires calibration)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size (default: 1)",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=1.0,
        help="Workspace size in GB (default: 1.0)",
    )

    args = parser.parse_args()

    fp16 = args.fp16 and not args.no_fp16

    export_to_tensorrt(
        onnx_path=args.onnx,
        output_path=args.output,
        fp16=fp16,
        int8=args.int8,
        max_batch_size=args.max_batch_size,
        workspace_size_gb=args.workspace,
    )


def main():
    """Main entry point."""
    convert_onnx_to_tensorrt_cli()


if __name__ == "__main__":
    main()
