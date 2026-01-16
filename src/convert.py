import os
from .model_loader import load_model

# Lazy import for ai_edge_torch because it might not be installed in inference env
# import ai_edge_torch

def convert_to_tflite(model_name='mobilenet_v2', output_dir='models'):
    """
    Convert a PyTorch model to LiteRT (TFLite) format.
    
    Args:
        model_name (str): Name of the model to convert.
        output_dir (str): Directory to save the converted model.
    """
    import torch
    import ai_edge_torch
    
    print(f"Converting {model_name} to LiteRT...")
    
    # 1. Load PyTorch model
    model = load_model(model_name, pretrained=True)
    model.eval()
    
    # 2. Prepare sample input (1, 3, 224, 224)
    sample_input = torch.randn(1, 3, 224, 224)
    
    # 3. Convert using ai-edge-torch
    print("Running ai_edge_torch.convert...")
    
    if os.environ.get("QUANTIZE_INT8"):
        print("Applying INT8 Quantization (Dynamic) using ai_edge_torch API...")
        from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
        from ai_edge_torch.quantize.quant_config import QuantConfig
        
        # Create quantizer and config
        # Use Dynamic Quantization for robustness in this demo
        pt2e_quantizer = PT2EQuantizer().set_global(get_symmetric_quantization_config(is_dynamic=True))
        q_config = QuantConfig(pt2e_quantizer=pt2e_quantizer)
        
        print("Converting with QuantConfig...")
        edge_model = ai_edge_torch.convert(model, (sample_input,), quant_config=q_config)
        model_name += "_int8"
    else:
        print("Converting FP32 model...")
        edge_model = ai_edge_torch.convert(model, (sample_input,))
    
    # 4. Save the model
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.tflite")
    edge_model.export(output_path)
    
    print(f"Conversion complete! Model saved to {output_path}")

if __name__ == "__main__":
    convert_to_tflite()
