
import os
import torch
import ai_edge_torch
import tensorflow as tf
import numpy as np
from .model_loader import load_model

def representative_dataset():
    """
    Generates a representative dataset for calibration.
    """
    # Use a small subset of real data or random if not available
    # Here we use random data for demonstration as in convert_onnx.py
    for _ in range(100):
        data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        yield [data]

def convert_aiedge_int8(model_name='mobilenet_v2', output_dir='models'):
    """
    Converts PyTorch -> SavedModel (via ai-edge-torch) -> TFLite INT8 (via TF Converter).
    This avoids the stablehlo errors in direct ai-edge-torch quantization.
    """
    print(f"Starting 'SavedModel Intercept' conversion for {model_name}...")
    
    # 1. Load PyTorch model
    model = load_model(model_name, pretrained=True)
    model.eval()
    
    sample_input = torch.randn(1, 3, 224, 224)
    
    # 2. Convert to TF SavedModel using ai-edge-torch
    # We use a temporary directory to store the SavedModel
    saved_model_path = os.path.join(output_dir, 'temp_saved_model')
    os.makedirs(saved_model_path, exist_ok=True)
    
    print(f"Exporting to SavedModel at {saved_model_path}...")
    try:
        # ai_edge_torch.convert normally returns a TFLite model, but we strictly want the SavedModel artifact.
        # correct usage: pass _saved_model_dir to convert()
        ai_edge_torch.convert(
            model,
            (sample_input,),
            _saved_model_dir=saved_model_path
        )
        print("Scrubbing ai-edge-torch output... SavedModel should operate now.")
    except Exception as e:
        print(f"Warning: ai_edge_torch.convert might have failed to finalize TFLite but SavedModel might be there. Error: {e}")
    
    # 3. Convert SavedModel to TFLite INT8 using standard TensorFlow Converter
    print("Converting SavedModel to TFLite INT8...")
    
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"SavedModel not found at {saved_model_path}")
        
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Standard INT8 Quantization Config
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Ensure full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8 # or tf.int8
    
    try:
        tflite_quant_model = converter.convert()
        
        # 4. Save the model
        output_path = os.path.join(output_dir, f"{model_name}_aiedge_int8.tflite")
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)
            
        print(f"Successfully created: {output_path}")
        print(f"Size: {len(tflite_quant_model) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Failed to convert SavedModel to TFLite INT8: {e}")
        raise e

if __name__ == '__main__':
    convert_aiedge_int8()
