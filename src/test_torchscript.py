"""
Test script to verify TorchScript model loading and inference.
"""

import torch
import numpy as np
import sys
import os

# Add src directory to path
src_dir = r"d:\Desktop\EHT\EHT_Analytica\src"
sys.path.insert(0, src_dir)

from EhtAnalytica.tracker.modelpacker import load_packed_model


def test_torchscript_model():
    """Test loading and running inference with a TorchScript model."""
    
    # Path to a TorchScript model
    model_path = r"d:\Desktop\EHT\EHT_Analytica\model\track_unet_v2\torchscript\P2_R0.pt"
    
    print(f"Loading TorchScript model: {model_path}")
    
    try:
        # Load the model
        model = load_packed_model(model_path)
        print("✓ Model loaded successfully!")
        
        # Create a dummy input (1 batch, 1 channel, 512x512)
        dummy_input = torch.randn(1, 1, 512, 512)
        print(f"\nInput shape: {dummy_input.shape}")
        
        # Run inference
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output value range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        print("\n✓ TorchScript model works correctly!")
        
        # Check file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\nModel file size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_torchscript_model()
