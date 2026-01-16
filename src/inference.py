import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time
import os
import urllib.request

def load_labels(filename="imagenet_labels.txt"):
    if not os.path.exists(filename):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        print(f"Downloading labels from {url}...")
        urllib.request.urlretrieve(url, filename)
    
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def preprocess_image(image_path, input_shape):
    """
    Preprocess image to match MobileNetV2 requirements:
    - Resize to 256
    - CenterCrop to 224
    - Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    img = Image.open(image_path).convert('RGB')
    
    # 1. Resize (maintain aspect ratio, min side 256)
    # PyTorch's Resize(256) resizes the smaller edge to 256
    width, height = img.size
    if width < height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))
        
    img = img.resize((new_width, new_height), Image.BILINEAR)
    
    # 2. CenterCrop 224
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # 3. To Tensor & Normalize
    input_data = np.array(img, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    input_data = (input_data - mean) / std
    
    # 4. Transpose to (1, 3, 224, 224) as PyTorch exports NCHW
    # BUT, ai-edge-torch might export as NHWC depending on config.
    # Usually ai-edge-torch respects the PyTorch layout if not optimized for TFLite GPU delegate.
    # However, standard TFLite models are NHWC. 
    # Let's inspect the model input details in the main function to be sure.
    # For now, let's assume NCHW because we exported a PyTorch model directly.
    # Wait, ai-edge-torch usually converts to TOSA/XNNPACK friendly format which might still look like NCHW or NHWC.
    # Standard MobileNetV2 TFLite from TF is NHWC.
    # Let's handle this dynamically in the inference function.
    
    # For now, return CHW (PyTorch default)
    input_data = input_data.transpose((2, 0, 1))
    
    return np.expand_dims(input_data, axis=0)

def run_inference(model_path, image_path):
    print(f"Loading LiteRT model: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check expected input shape
    input_shape = input_details[0]['shape']
    print(f"Model input shape: {input_shape}")
    
    # Preprocess
    input_data = preprocess_image(image_path, input_shape)
    
    # Handle NCHW vs NHWC automagically if possible, or just expect NCHW since it's from PyTorch
    if input_shape[1] == 224 and input_shape[2] == 224 and input_shape[3] == 3:
        # Model expects NHWC
        print("Model expects NHWC input. Transposing...")
        input_data = input_data.transpose((0, 2, 3, 1))
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get Top-5
    labels = load_labels()
    # Flatten result
    results = np.squeeze(output_data)
    top_k = results.argsort()[-5:][::-1]
    
    inference_time = (end_time - start_time) * 1000
    print(f"\nInference Time: {inference_time:.2f} ms")
    
    top_results = []
    print("Top 5 Results:")
    for i in top_k:
        score = results[i]
        label = labels[i]
        print(f"{label}: {score:.4f}")
        top_results.append({"label": label, "score": float(score)})
        
    return {
        "inference_time_ms": inference_time,
        "top_results": top_results,
        "input_data": input_data # Return for visualization if needed
    }

if __name__ == "__main__":
    # Create a dummy image if not exists
    if not os.path.exists("test_image.jpg"):
        from PIL import ImageDraw
        img = Image.new('RGB', (300, 300), color = 'red')
        img.save('test_image.jpg')
        
    run_inference("models/mobilenet_v2.tflite", "test_image.jpg")
