import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
import resource
import os
import glob
from tabulate import tabulate


def get_model_size(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)


def benchmark_model(model_path, num_runs=50):
    print(f"Benchmarking {model_path}...")
    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Generate random input based on dtype
    if input_dtype == np.uint8:
        input_data = np.random.randint(0, 255, input_shape).astype(np.uint8)
    elif input_dtype == np.int8:
        input_data = np.random.randint(-128, 127, input_shape).astype(np.int8)
    else:
        input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

    # Benchmark
    latencies = []

    # Measure memory before
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(num_runs):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)

    # Measure memory after
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "file": os.path.basename(model_path),
        "size_mb": get_model_size(model_path),
        "avg_ms": np.mean(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "std_ms": np.std(latencies),
        "mem_peak_mb": (mem_after) / 1024,  # maxrss is in KB on Linux
    }


def benchmark_compiled_model(model_path, num_runs=50):
    print(f"Benchmarking CompiledModel {model_path}...")
    try:
        model = CompiledModel.from_file(model_path)
    except Exception as e:
        print(f"Failed to load CompiledModel: {e}")
        return None

    signature_list = model.get_signature_list()
    # Assume the first signature
    if not signature_list:
        print("No signatures found in CompiledModel")
        return None
    signature_key = list(signature_list.keys())[0]

    # Generate random input
    # We need to know shape. CompiledModel doesn't expose it easily.
    # Let's assume (1, 224, 224, 3) float32 for now as we are targeting MobileNetV2
    # In a real generic benchmark we would need to inspect signature inputs more deeply or use interpreter.
    input_size = (1, 224, 224, 3)
    input_data = np.random.randn(*input_size).astype(np.float32)

    # Create Buffers
    input_map = {}
    inputs = signature_list[signature_key]["inputs"]
    for input_name in inputs:
        tensor_buffer = model.create_input_buffer_by_name(signature_key, input_name)
        try:
            tensor_buffer.write(input_data)
        except Exception as e:
            # Fallback for quantized models
            # usually int8 for modern TFLite, or uint8 for legacy.
            # However, ai_edge_litert wrapper might only support int8.
            print(
                f"Warning: Write failed with default float32 data. Retrying with int8. Error: {e}"
            )
            # Try int8
            try:
                input_data_int8 = (
                    (input_data * 255 - 128).clip(-128, 127).astype(np.int8)
                )
                # Approximation: float input is roughly N(0,1), but for benchmark random is fine.
                # Just need valid buffer write.
                tensor_buffer.write(input_data_int8)
            except Exception as e2:
                print(f"Failed to write input data even with int8: {e2}")
                # Could try uint8 but we know it throws ValueError in current version
                return None
        input_map[input_name] = tensor_buffer

    output_map = {}
    outputs = signature_list[signature_key]["outputs"]
    for output_name in outputs:
        output_map[output_name] = model.create_output_buffer_by_name(
            signature_key, output_name
        )

    # Warmup
    for _ in range(5):
        model.run_by_name(signature_key, input_map, output_map)

    # Benchmark
    latencies = []

    # Measure memory before
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(num_runs):
        start_time = time.time()
        model.run_by_name(signature_key, input_map, output_map)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)

    # Measure memory after
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "file": os.path.basename(model_path) + " (Compiled)",
        "size_mb": get_model_size(model_path),
        "avg_ms": np.mean(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "std_ms": np.std(latencies),
        "mem_peak_mb": (mem_after) / 1024,
    }


def run_comparison(models_dir=None):
    if models_dir is None:
        # Resolve models dir relative to this script: src/benchmark.py -> ../models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")

    print(f"Searching for models in: {os.path.abspath(models_dir)}")
    model_files = glob.glob(os.path.join(models_dir, "*.tflite"))
    if not model_files:
        print("No .tflite models found in", models_dir)
        return

    results = []
    print(f"Found {len(model_files)} models. Starting benchmark...\n")

    for model_file in sorted(model_files):
        # 1. Interpreter Benchmark
        res_interp = benchmark_model(model_file)
        if res_interp:
            # Mark it clearly
            res_interp["file"] = os.path.basename(model_file) + " (Interpreter)"
            results.append(res_interp)

        # 2. CompiledModel Benchmark
        res_compiled = benchmark_compiled_model(model_file)
        if res_compiled:
            results.append(res_compiled)

    # Print Table
    headers = [
        "Model",
        "Size (MB)",
        "Avg Latency (ms)",
        "Min (ms)",
        "Max (ms)",
        "Peak Mem (MB)",
    ]
    table_data = []
    for r in results:
        table_data.append(
            [
                r["file"],
                f"{r['size_mb']:.2f}",
                f"{r['avg_ms']:.2f}",
                f"{r['min_ms']:.2f}",
                f"{r['max_ms']:.2f}",
                f"{r['mem_peak_mb']:.2f}",
            ]
        )

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 60)

    return results


if __name__ == "__main__":
    # uv add tabulate to use this prettily
    run_comparison()
