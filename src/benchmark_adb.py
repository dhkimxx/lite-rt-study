import os
import subprocess
import re
from tabulate import tabulate

def run_command(cmd):
    """Runs a shell command and returns the output."""
    try:
        # Merge stderr into stdout because benchmark_model logs to stderr frequently
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(e.stderr)
        return None

def setup_adb_env(bin_path, models_dir):
    """Pushes the benchmark binary and models to the device."""
    print("Setting up ADB environment...")
    
    # 1. Create remote dir
    run_command("adb shell mkdir -p /data/local/tmp/lite_rt")
    
    # 2. Push benchmark binary
    print(f"Pushing benchmark binary: {bin_path}")
    run_command(f"adb push {bin_path} /data/local/tmp/lite_rt/benchmark_model")
    run_command("adb shell chmod +x /data/local/tmp/lite_rt/benchmark_model")
    
    # 3. Push models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.tflite')]
    for model_file in model_files:
        src_path = os.path.join(models_dir, model_file)
        print(f"Pushing model: {model_file}")
        run_command(f"adb push {src_path} /data/local/tmp/lite_rt/{model_file}")

    return model_files

def run_adb_benchmark(model_name, use_gpu=False, use_npu=False):
    """Runs the benchmark for a specific model on the device."""
    base_cmd = f"adb shell /data/local/tmp/lite_rt/benchmark_model --graph=/data/local/tmp/lite_rt/{model_name} --num_runs=50"
    
    if use_gpu:
        base_cmd += " --use_gpu=true"
    if use_npu:
        base_cmd += " --use_npu=true"
        
    print(f"Running benchmark for {model_name} (GPU={use_gpu}, NPU={use_npu})...")
    output = run_command(base_cmd)
    
    if not output:
        return None
    
    # Parse output using regex
    # Example: INFO: [benchmark_litert_model.h:88] Inference (avg):      71.45 ms (14 runs)
    match = re.search(r"Inference \(avg\):\s+([\d.]+) ms", output)
    if match:
        latency_ms = float(match.group(1))
        return latency_ms
            
    print(f"Failed to parse output for {model_name}")
    return None

def main():
    bin_path = "bin/benchmark_model_android_arm64"
    models_dir = "models"
    
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found. Please download it first.")
        return

    model_files = setup_adb_env(bin_path, models_dir)
    
    results = []
    
    for model_file in sorted(model_files):
        # 1. Run on CPU
        cpu_latency = run_adb_benchmark(model_file)
        
        # 2. Run on GPU
        gpu_latency = run_adb_benchmark(model_file, use_gpu=True)
        
        # 3. Run on NPU
        npu_latency = run_adb_benchmark(model_file, use_npu=True)
        
        results.append({
            "Model": model_file,
            "CPU (ms)": f"{cpu_latency:.2f}" if cpu_latency else "N/A",
            "GPU (ms)": f"{gpu_latency:.2f}" if gpu_latency else "N/A",
            "NPU (ms)": f"{npu_latency:.2f}" if npu_latency else "N/A"
        })

    # Print results
    print("\n" + "="*80)
    print("ANDROID (ADB) BENCHMARK RESULTS")
    print("="*80)
    headers = ["Model", "CPU (ms)", "GPU (ms)", "NPU (ms)"]
    table_data = [[r["Model"], r["CPU (ms)"], r["GPU (ms)"], r["NPU (ms)"]] for r in results]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80)
    print("\n[NOTE] INT8 models usually perform best with NPU on supported hardware.")

if __name__ == "__main__":
    main()
