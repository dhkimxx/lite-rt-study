import time
import numpy as np
import tflite_runtime.interpreter as tflite
import resource
import os
import glob
from tabulate import tabulate

def get_model_size(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)

def benchmark_model(model_path, num_runs=50):
    print(f"Benchmarking {model_path}...")
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    # Generate random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
    # Benchmark
    latencies = []
    
    # Measure memory before
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    for _ in range(num_runs):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
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
        "mem_peak_mb": (mem_after) / 1024 # maxrss is in KB on Linux
    }

def run_comparison(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "*.tflite"))
    if not model_files:
        print("No .tflite models found in", models_dir)
        return

    results = []
    print(f"Found {len(model_files)} models. Starting benchmark...\n")
    
    for model_file in sorted(model_files):
        res = benchmark_model(model_file)
        if res:
            results.append(res)
            
    # Print Table
    headers = ["Model", "Size (MB)", "Avg Latency (ms)", "Min (ms)", "Max (ms)", "Peak Mem (MB)"]
    table_data = []
    for r in results:
        table_data.append([
            r["file"],
            f"{r['size_mb']:.2f}",
            f"{r['avg_ms']:.2f}",
            f"{r['min_ms']:.2f}",
            f"{r['max_ms']:.2f}",
            f"{r['mem_peak_mb']:.2f}"
        ])
        
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*60)
    
    return results

if __name__ == "__main__":
    # uv add tabulate to use this prettily
    run_comparison()
