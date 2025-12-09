"""
ONNX export for lane detection models.
"""

import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_size: int = 224,
    opset_version: int = 13,
    dynamic_batch: bool = True,
    verify: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path for the output ONNX file
        input_size: Input image size
        opset_version: ONNX opset version
        dynamic_batch: Whether to allow dynamic batch size
        verify: Whether to verify the exported model
        verbose: Print progress information

    Returns:
        Path to the exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Define dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    if verbose:
        print(f"Exporting model to ONNX: {output_path}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if verbose:
        print(f"Model exported successfully")

    # Verify the model
    if verify:
        if verbose:
            print("Verifying ONNX model...")

        # Check model is valid
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(str(output_path))

        # Run inference
        input_name = ort_session.get_inputs()[0].name
        ort_output = ort_session.run(
            None,
            {input_name: dummy_input.numpy()},
        )[0]

        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = model(dummy_input).numpy()

        # Check outputs are close
        np.testing.assert_allclose(
            torch_output,
            ort_output,
            rtol=1e-3,
            atol=1e-5,
        )

        if verbose:
            print("Verification passed! ONNX model matches PyTorch output.")

    # Print model info
    if verbose:
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"Model size: {file_size:.2f} MB")

    return output_path


def load_checkpoint_and_export(
    checkpoint_path: str | Path,
    output_path: str | Path,
    architecture: str = "mobilenetv3",
    input_size: int = 224,
    verbose: bool = True,
) -> Path:
    """
    Load a training checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        output_path: Path for the output ONNX file
        architecture: Model architecture name
        input_size: Input image size
        verbose: Print progress

    Returns:
        Path to the exported ONNX file
    """
    from src.models.factory import create_model
    from src.models.base import ModelConfig

    checkpoint_path = Path(checkpoint_path)

    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create model
    config = ModelConfig(input_size=input_size)
    model = create_model(architecture, pretrained=False, config=config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Export
    return export_to_onnx(
        model,
        output_path,
        input_size=input_size,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mobilenetv3",
        help="Model architecture",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification",
    )

    args = parser.parse_args()

    load_checkpoint_and_export(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        architecture=args.architecture,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()
