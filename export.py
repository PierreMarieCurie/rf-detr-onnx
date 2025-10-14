# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 PierreMarieCurie
#
# Portions of this file use RF-DETR
# Copyright (c) Roboflow
# Licensed under the Apache License, Version 2.0
# See https://www.apache.org/licenses/LICENSE-2.0 for details.
# ------------------------------------------------------------------------

import argparse
import torch
from pathlib import Path
from typing import Optional
from rfdetr.detr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRSmall,
    RFDETRNano,
    RFDETRMedium,
    RFDETRSegPreview
)
import onnx
from onnxsim import simplify

def load_rfdetr_model(cpkt: str):
    try:
        # Load checkpoint
        print(f"Loading checkpoint: {cpkt}...")
        obj = torch.load(cpkt , weights_only=False)
        args = obj.get("args", None)
        
        # Object detection model
        if args and hasattr(args, "resolution") and hasattr(args, "hidden_dim"):
            resolution = args.resolution
            if resolution == 384:
                print("Assuming the checkpoint corresponds to RF-DETR Nano")
                return RFDETRNano(pretrain_weights=cpkt)
            elif resolution == 512:
                print("Assuming the checkpoint corresponds to RF-DETR Small")
                return RFDETRSmall(pretrain_weights=cpkt)
            elif resolution == 576:
                print("Assuming the checkpoint corresponds to RF-DETR Medium")
                return RFDETRMedium(pretrain_weights=cpkt)
            elif resolution == 560:
                hidden_dim = args.hidden_dim
                if hidden_dim == 256:
                    print("Assuming the checkpoint corresponds to RF-DETR Base")
                    return RFDETRBase(pretrain_weights=cpkt)
                elif hidden_dim == 384:
                    print("Assuming the checkpoint corresponds to RF-DETR Large")
                    return RFDETRLarge(pretrain_weights=cpkt)
                else:
                    raise ValueError(f"Unknown hidden_dim: {hidden_dim}")
            else:
                raise ValueError(f"Unknown resolution: {resolution}")
        
        # Instance segmentation model
        elif "model" in obj:
            if len(obj["model"]) == 544:
                print("Assuming the checkpoint corresponds to RF-DETR with segmentation head")
                return RFDETRSegPreview(pretrain_weights=cpkt)
            else:
                raise ValueError(f"Unknown model architecture")
    except Exception as e:
        raise RuntimeError(f"Error loading RF-DETR model from checkpoint '{cpkt}'") from e

def export_onnx(checkpoint: str, onnx_name: Optional[str] = None, no_simplify: bool = False) -> None:
    """
    Export an RF-DETR model to ONNX format.

    Args:
        checkpoint (str): Path to the model checkpoint file (e.g. "rf-detr-nano.pth").
        onnx_name (str): Name of the output ONNX file.
    """
    try:
        # Validate inputs
        if not isinstance(checkpoint, str):
            raise TypeError(f"Expected 'checkpoint' to be a str, got {type(checkpoint).__name__}")
        path = Path(checkpoint)
        if path.suffix not in {".pt", ".pth"}:
            raise ValueError(f"Invalid model file: {checkpoint}. Expected a .pt or .pth file.")
        if onnx_name is None:
            onnx_name = path.with_suffix(".onnx").name

        # Load RF-DETR model
        rfdetr_model = load_rfdetr_model(checkpoint)
        model = rfdetr_model.model.model
        config = rfdetr_model.model_config

        # Forward pass to get output layer names
        resolution = config.resolution
        device = config.device
        dummy_input = torch.randn(1, 3, resolution, resolution, device=device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        if len(output) == 4: # Object detection
            output_names = list(output.keys())[:2][::-1]
        elif len(output) == 5: # Instance segmentation
            output_names = list(output.keys())[:3][::-1]
        else:
            raise ValueError("Unexpected model output structure")

        # Export to ONNX
        print(f"Exporting model to ONNX: {onnx_name}...")
        model.export()
        torch.onnx.export(
            model,
            dummy_input,
            onnx_name,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names
        )
        
        # Simplify ONNX model
        if not no_simplify:
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(onnx_name)
            model_simplified, check = simplify(onnx_model)
            if check:
                onnx.save(model_simplified, onnx_name)
            else:
                raise RuntimeError("ONNX simplification check failed")
        
        print(f"Model successfully exported to {onnx_name}")

    except Exception as e:
        raise RuntimeError(f"Failed to export model from checkpoint '{checkpoint}'") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export RF-DETR model to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a custom RF-DETR checkpoint (.pth for object detection, .pt for instance segmentation)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the output ONNX model file. If not provided, checkpoint name will be used with .onnx extension",
    )
    parser.add_argument(
    "--no-simplify",
    action="store_true",
    help="Do not simplify the ONNX model (diseabled by default)",
    )
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.model_name, args.no_simplify)