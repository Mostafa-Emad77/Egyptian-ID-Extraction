import copy
from models.CRDN import (
    UNetRNN, VGG16RNN, ResNet18RNN, ResNet34RNN, 
    ResNet50RNN, ResNet101RNN, ResNet152RNN, 
    ResNet50UNet, ResNet50FCN
)


def get_model(model_dict, n_classes):
    """
    Get a model instance based on the provided configuration.
    
    Args:
        model_dict (dict): Dictionary containing model configuration
        n_classes (int): Number of output classes
        
    Returns:
        model: PyTorch model instance
        
    Raises:
        ValueError: If model architecture is not available
    """
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    try:
        model = model(
            input_channel=3,
            n_classes=n_classes,
            kernel_size=3,
            feature_scale=4,
            decoder="vanilla",
            bias=True
        )
        return model
    except Exception as e:
        raise ValueError(f"Failed to initialize model {name}: {str(e)}")


def _get_model_instance(name):
    """
    Get model class by name.
    
    Args:
        name (str): Name of the model architecture
        
    Returns:
        model_class: Model class
        
    Raises:
        ValueError: If model architecture is not available
    """
    available_models = {
        "UNetRNN": UNetRNN,
        "VGG16RNN": VGG16RNN,
        "ResNet18RNN": ResNet18RNN,
        "ResNet34RNN": ResNet34RNN,
        "ResNet50RNN": ResNet50RNN,
        "ResNet101RNN": ResNet101RNN,
        "ResNet152RNN": ResNet152RNN,
        "ResNet50UNet": ResNet50UNet,
        "ResNet50FCN": ResNet50FCN
    }
    
    if name not in available_models:
        raise ValueError(f"Model '{name}' not available. Available models: {', '.join(available_models.keys())}")
    
    return available_models[name]

