import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
from PIL import Image
import time
import os
import urllib.request


def load_labels(filename="imagenet_labels.txt"):
    if not os.path.exists(filename):
        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
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
    img = Image.open(image_path).convert("RGB")

    # 1. Resize (maintain aspect ratio, min side 256)
    # PyTorch's Resize(256) resizes the smaller edge to 256
    width, height = img.size
    if width < height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))

    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

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
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check expected input shape
    input_shape = input_details[0]["shape"]
    print(f"Model input shape: {input_shape}")

    # Preprocess
    input_data = preprocess_image(image_path, input_shape)

    # Handle NCHW vs NHWC automagically if possible, or just expect NCHW since it's from PyTorch
    if input_shape[1] == 224 and input_shape[2] == 224 and input_shape[3] == 3:
        # Model expects NHWC
        print("Model expects NHWC input. Transposing...")
        input_data = input_data.transpose((0, 2, 3, 1))

    interpreter.set_tensor(input_details[0]["index"], input_data)

    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]["index"])

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
        "input_data": input_data,  # Return for visualization if needed
    }


def run_inference_compiled(model_path, image_path):
    print(f"Loading CompiledModel: {model_path}")
    try:
        model = CompiledModel.from_file(model_path)
    except Exception as e:
        print(f"Failed to load CompiledModel: {e}")
        return None

    # Get signature
    signature_list = model.get_signature_list()
    # Assume the first signature if multiple or specific key if known
    # Typical TFLite signatures: 'serving_default'
    signature_key = list(signature_list.keys())[0]
    print(f"Using signature: {signature_key}")

    # Preprocess
    # We assume 224x224 input as standard for these models
    input_shape = (1, 3, 224, 224)
    input_data = preprocess_image(image_path, input_shape)

    # MobileNetV2 usually uses NHWC
    # However, this specific model seems to use NCHW based on run_inference output [1, 3, 224, 224]
    # if input_data.shape == (1, 3, 224, 224):
    #    input_data = input_data.transpose((0, 2, 3, 1))

    # Create Buffers
    input_map = {}
    inputs = signature_list[signature_key]["inputs"]

    for input_name in inputs:
        # Create buffer
        tensor_buffer = model.create_input_buffer_by_name(signature_key, input_name)
        # Write data (expecting numpy array)
        try:
            tensor_buffer.write(input_data)
        except Exception as e:
            print(
                f"Warning: Write failed with default data. Retrying with uint8 quantization. Error: {e}"
            )
            # Re-read raw image for simple quantization if needed, OR just cast.
            # Normalization logic was specific to float model.
            # For uint8 TFLite, typically input is 0-255 RGB.
            # We already have normalized float data (-2..2 approx).
            # Simply casting that to uint8 is WRONG (it becomes 0, 1, 2).
            # We need to ideally undo normalization or just use un-normalized data.
            # But we don't have access to original img easily here without re-reading.
            # Let's approximate: un-normalize then scale.
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # input_data = (orig/255 - mean) / std
            # orig/255 = input_data * std + mean
            # orig = (input_data * std + mean) * 255
            mean = (
                np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
                if input_data.shape[-1] == 3
                else np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
                    1, 3, 1, 1
                )
            )
            std = (
                np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
                if input_data.shape[-1] == 3
                else np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(
                    1, 3, 1, 1
                )
            )

            # Note: input_data might be NCHW or NHWC.
            # If line 147 transposed it to NHWC, shape is (1, 224, 224, 3).
            # We need to make sure mean/std broadcasting works.

            # Int8 quantization usually uses range -128 to 127.
            # 0..255 -> -128..127 : subtract 128
            input_data_recovered = (input_data * std + mean) * 255.0
            input_int8 = (input_data_recovered - 128).clip(-128, 127).astype(np.int8)

            try:
                tensor_buffer.write(input_int8)
            except Exception as e2:
                print(f"Failed to write input data even with int8: {e2}")
                return None

        input_map[input_name] = tensor_buffer

    output_map = {}
    outputs = signature_list[signature_key]["outputs"]
    for output_name in outputs:
        output_map[output_name] = model.create_output_buffer_by_name(
            signature_key, output_name
        )

    start_time = time.time()
    model.run_by_name(signature_key, input_map, output_map)
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000
    print(f"\nCompiledModel Inference Time: {inference_time:.2f} ms")

    # Process Output
    output_name = outputs[0]
    output_buffer = output_map[output_name]

    # Read output
    # Assuming standard classification output size 1001 for MobileNetV2
    # But some models are 1000.
    output_size = 1001
    try:
        results = output_buffer.read(output_size, np.float32)
    except RuntimeError:
        # Retry with 1000
        output_size = 1000
        print(f"Warning: Failed to read 1001 outputs, trying {output_size}...")
        results = output_buffer.read(output_size, np.float32)

    # Get Top-5
    labels = load_labels()
    results = np.squeeze(results)

    # Safety check for label mapping
    if results.size > len(labels):
        results = results[: len(labels)]

    top_k = results.argsort()[-5:][::-1]

    top_results = []
    print("Top 5 Results (Compiled):")
    for i in top_k:
        score = results[i]
        label = labels[i] if i < len(labels) else str(i)
        print(f"{label}: {score:.4f}")
        top_results.append({"label": label, "score": float(score)})

    return {"inference_time_ms": inference_time, "top_results": top_results}


if __name__ == "__main__":
    # Create a dummy image if not exists
    if not os.path.exists("test_image.jpg"):
        from PIL import ImageDraw

        img = Image.new("RGB", (300, 300), color="red")
        img.save("test_image.jpg")

    print("--- Interpreter API ---")
    run_inference("models/mobilenet_v2.tflite", "test_image.jpg")
    print("\n--- CompiledModel API ---")
    run_inference_compiled("models/mobilenet_v2.tflite", "test_image.jpg")
