import os
import sys
import torch
from models import get_model
from collections import OrderedDict
from typing import Tuple, Optional

# Default model configuration
DEFAULT_MODEL_ARCH = 'UNetRNN'
DEFAULT_MODEL_PATH = "CRDN1000.pkl"

def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    model_arch: str = DEFAULT_MODEL_ARCH,
    device_id: Optional[int] = 0
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a trained model from disk and prepare it for inference.
    
    Args:
        model_path (str): Path to the saved model weights
        model_arch (str): Name of the model architecture to use
        device_id (int, optional): GPU device ID to use. Defaults to 0.
        
    Returns:
        Tuple[torch.nn.Module, torch.device]: Loaded model and device it's on
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
        ValueError: If invalid device ID is provided
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Setup device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU device ID: {device_id}. Available devices: {torch.cuda.device_count()}")
        device = torch.device(f"cuda:{device_id}")
    
    try:
        # Initialize model
        model = get_model({'arch': model_arch}, n_classes=2)
        model = model.to(device)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        if "model_state" not in state_dict:
            raise RuntimeError("Invalid model file format: missing 'model_state' key")
            
        state = convert_state_dict(state_dict["model_state"])
        model.load_state_dict(state)
        
        # Set to evaluation mode
        model.eval()
        
        return model, device
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def convert_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """
    Convert a state dict saved from a DataParallel module to normal module state dict.
    
    Args:
        state_dict (OrderedDict): State dict from DataParallel model
        
    Returns:
        OrderedDict: Converted state dict
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict
        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        item_name = k[7:]  # remove `module.`
        new_state_dict[item_name] = v
    return new_state_dict

if __name__ == "__main__":
    try:
        trained_model, device = load_model()
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
