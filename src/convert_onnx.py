import torch
import onnx
import os
import subprocess
from .model_loader import load_model

def convert_to_onnx(model_name='mobilenet_v2', output_dir='models'):
    print(f"Converting {model_name} to ONNX...")
    
    # 1. Load PyTorch model
    model = load_model(model_name, pretrained=True)
    model.eval()
    
    # 2. Prepare sample input
    sample_input = torch.randn(1, 3, 224, 224)
    
    # 3. Export to ONNX
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified.")
    
    return onnx_path

def convert_onnx_to_tflite(onnx_path, output_dir='models'):
    print(f"Converting {onnx_path} to TFLite (INT8) using onnx2tf...")
    
    # onnx2tf command
    # -i: Input ONNX path
    # -o: Output directory
    # -oiqt: Output Integer Quantized TFLite (Full INT8)
    # -qt: Quantization Type (per-channel or per-tensor, usually per-channel is default)
    # But wait, onnx2tf handles quantization differently.
    # It usually generates FP32 by default.
    # To get INT8, we can use the -oiqt flag (Output INT8 Quantized TFLite).
    # NOTE: onnx2tf's quantization support might be experimental or require calibration data.
    # onnx2tf usually needs a calibration data file for full integer quantization.
    # Let's try converting to FP32 TFLite first using onnx2tf (it's very good at structure),
    # then use TFLiteConverter (TensorFlow) to quantize it?
    # Actually, onnx2tf's strength is generating a valid SavedModel/TFLite structure.
    # The best 'Legacy' flow is:
    # 1. onnx2tf -> SavedModel (tf)
    # 2. tf.lite.TFLiteConverter -> TFLite INT8 (using calibration)
    
    # But onnx2tf outputs tflite directly.
    # Let's check if we can simply use onnx2tf to generate SavedModel (-osd).
    
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    saved_model_dir = os.path.join(output_dir, f"{model_name}_saved_model")
    
    # Use onnx2tf via subprocess
    cmd = [
        "onnx2tf",
        "-i", onnx_path,
        "-o", saved_model_dir,
        "-osd" # Output SavedModel Directory
    ]
    
    print("Running onnx2tf:", " ".join(cmd))
    subprocess.check_call(cmd)
    
    return saved_model_dir

def quantize_saved_model(saved_model_dir, output_dir='models'):
    print("Quantizing SavedModel to TFLite INT8...")
    import tensorflow as tf
    import numpy as np
    from .data_loader import get_calibration_loader
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Calibration
    def representative_dataset():
        loader = get_calibration_loader()
        for i, batch in enumerate(loader):
            # batch is torch tensor (N, 3, 224, 224)
            # TFLite expects numpy array (N, 224, 224, 3) for NHWC usually?
            # onnx2tf usually handles NCHW -> NHWC conversion.
            # Convert torch tensor to numpy
            data = batch.numpy()
            # Transpose if needed? 
            # onnx2tf usually converts to NHWC. Let's check input details if possible, 
            # or just provide NHWC which is standard for TF.
            data = np.transpose(data, (0, 2, 3, 1)) # NCHW -> NHWC
            yield [data]

    converter.representative_dataset = representative_dataset
    
    # Full Integer Quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8 # Or tf.int8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    output_path = os.path.join(output_dir, "mobilenet_v2_legacy_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"INT8 Model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    onnx_path = convert_to_onnx()
    saved_model_dir = convert_onnx_to_tflite(onnx_path)
    quantize_saved_model(saved_model_dir)
