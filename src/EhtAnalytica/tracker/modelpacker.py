"""
Model packing utilities for converting PyTorch models to TorchScript format.
"""

import torch
import os


def pack_model(model, name, output_dir=None):
    """
    Save a PyTorch model as a TorchScript file.
    
    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to save
    name : str
        Name for the output file (without extension)
    output_dir : str, optional
        Directory to save the TorchScript file. If None, saves in current directory
    
    Returns
    -------
    str
        Path to the saved TorchScript file
    """
    # Set model to evaluation mode
    model.eval()
    
    # Convert model to TorchScript using scripting
    try:
        scripted_model = torch.jit.script(model)
    except Exception as e:
        print(f"torch.jit.script failed: {e}")
        print("Attempting to use torch.jit.trace instead...")
        
        # If scripting fails, try tracing with a dummy input
        # Assuming input is 512x512 single channel image
        dummy_input = torch.randn(1, 1, 512, 512)
        try:
            scripted_model = torch.jit.trace(model, dummy_input)
        except Exception as trace_error:
            raise RuntimeError(f"Both torch.jit.script and torch.jit.trace failed: {trace_error}")
    
    # Prepare output path
    if output_dir is None:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Add .pt extension if not present
    if not name.endswith('.pt'):
        name = f"{name}.pt"
    
    output_path = os.path.join(output_dir, name)
    
    # Save the TorchScript model
    scripted_model.save(output_path)
    
    print(f"Model saved as TorchScript: {output_path}")
    
    return output_path


def load_packed_model(path):
    """
    Load a TorchScript model.
    
    Parameters
    ----------
    path : str
        Path to the TorchScript file
    
    Returns
    -------
    torch.jit.ScriptModule
        Loaded TorchScript model
    """
    model = torch.jit.load(path)
    model.eval()
    return model
