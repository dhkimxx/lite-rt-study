import torch
import torchvision.models as models
import os

def load_model(model_name='mobilenet_v2', pretrained=True, save_path=None):
    """
    Load a pre-trained model from torchvision.
    
    Args:
        model_name (str): Name of the model to load.
        pretrained (bool): Whether to load pre-trained weights.
        save_path (str, optional): Path to save the model state dict (optional).
        
    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
    """
    print(f"Loading {model_name} (pretrained={pretrained})...")
    
    if model_name == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    return model

if __name__ == "__main__":
    load_model(save_path="models/mobilenet_v2.pth")
