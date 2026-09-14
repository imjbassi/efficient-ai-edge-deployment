import os
import sys
import time
import json
import argparse
import math
import platform
import hashlib

# Try importing dependencies for actual benchmarking
try:
    import numpy as np
except ImportError:
    np = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import psutil
except ImportError:
    psutil = None

# Reference profile supplied with the repository. These are emulated/reference
# values and must not be described as fresh physical measurements.
REFERENCE_PROFILE = {
    "fp32_baseline": {
        "model_name": "MobileNetV2 (FP32 Baseline)",
        "latency_ms": 220.0,
        "latency_p90_ms": 245.0,
        "throughput_fps": 4.55,
        "ram_mb": 240.0,
        "model_size_mb": 13.6,
        "avg_power_w": 3.2,
        "energy_j": 0.704,  # 0.220s * 3.2W
        "energy_efficiency_ips_w": 1.42,  # 1.0 / 0.704
        "measurement_status": "reference_emulated"
    },
    "int8_quantized": {
        "model_name": "MobileNetV2 (INT8 Quantized TFLite)",
        "latency_ms": 95.0,
        "latency_p90_ms": 105.0,
        "throughput_fps": 10.53,
        "ram_mb": 65.0,
        "model_size_mb": 3.3,
        "avg_power_w": 3.0,
        "energy_j": 0.285,  # 0.095s * 3.0W
        "energy_efficiency_ips_w": 3.51,  # 1.0 / 0.285
        "measurement_status": "reference_emulated"
    }
}

def get_process_memory():
    """Returns current process RAM usage in MB."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0

def run_actual_benchmark(num_runs=100):
    """
    Runs actual CPU benchmarks of Keras FP32 and Quantized INT8 models.
    Only active if TensorFlow and NumPy are available.
    """
    print("\n" + "="*60)
    print("RUNNING LIVE LOCAL BENCHMARK ON CURRENT HOST CPU")
    print("="*60)

    # Check if files exist or generate them
    fp32_path = "models/mobilenet_v2.h5"
    int8_path = "models/mobilenet_v2_int8.tflite"

    os.makedirs("models", exist_ok=True)

    # Import factory elements to create models if needed
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from model_factory import get_mobilenet_v2
    from quantization import apply_int8_quantization

    # 1. FP32 Model Preparation
    if not os.path.exists(fp32_path):
        print(f"Baseline model not found. Generating Keras MobileNetV2...")
        model = get_mobilenet_v2()
        model.save(fp32_path)
    else:
        print(f"Loading existing FP32 model from {fp32_path}...")
        model = tf.keras.models.load_model(fp32_path)

    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)

    # 2. INT8 Model Preparation
    if not os.path.exists(int8_path):
        print(f"INT8 TFLite model not found. Quantizing model...")
        apply_int8_quantization(model, int8_path)

    int8_size = os.path.getsize(int8_path) / (1024 * 1024)

    # 3. Live RAM baseline
    ram_before = get_process_memory()

    # 4. FP32 Benchmark
    print(f"\nBenchmarking FP32 Baseline ({num_runs} runs)...")
    dummy_input = np.random.default_rng(42).uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)

    # Warm up
    _ = model.predict(dummy_input, verbose=0)

    latencies_fp32 = []
    ram_during_fp32 = []

    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        end_time = time.perf_counter()
        latencies_fp32.append((end_time - start_time) * 1000.0)
        ram_during_fp32.append(get_process_memory())

    mean_latency_fp32 = sum(latencies_fp32) / len(latencies_fp32)
    p90_latency_fp32 = float(np.percentile(latencies_fp32, 90))
    throughput_fp32 = 1000.0 / mean_latency_fp32
    ram_used_fp32 = max(ram_during_fp32)

    # 5. INT8 Benchmark
    print(f"Benchmarking INT8 Quantized TFLite ({num_runs} runs)...")
    interpreter = tf.lite.Interpreter(model_path=int8_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Prepare INT8 input using the model's actual quantization metadata.
    tflite_input_type = input_details['dtype']
    if tflite_input_type in (np.int8, np.uint8):
        scale, zero_point = input_details["quantization"]
        if not scale:
            raise ValueError("INT8 model has invalid input quantization metadata")
        info = np.iinfo(tflite_input_type)
        dummy_input_int8 = np.clip(np.round(dummy_input / scale + zero_point), info.min, info.max).astype(tflite_input_type)
    else:
        dummy_input_int8 = dummy_input

    interpreter.set_tensor(input_details['index'], dummy_input_int8)
    interpreter.invoke()  # One untimed warm-up, matching the Keras path.
    latencies_int8 = []
    ram_during_int8 = []

    for _ in range(num_runs):
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details['index'], dummy_input_int8)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details['index'])
        end_time = time.perf_counter()
        latencies_int8.append((end_time - start_time) * 1000.0)
        ram_during_int8.append(get_process_memory())

    mean_latency_int8 = sum(latencies_int8) / len(latencies_int8)
    p90_latency_int8 = float(np.percentile(latencies_int8, 90))
    throughput_int8 = 1000.0 / mean_latency_int8
    ram_used_int8 = max(ram_during_int8)

    # Simulated energy calculation (using power figures from paper for edge equivalence)
    power_fp32 = REFERENCE_PROFILE["fp32_baseline"]["avg_power_w"]
    power_int8 = REFERENCE_PROFILE["int8_quantized"]["avg_power_w"]

    energy_fp32 = (mean_latency_fp32 / 1000.0) * power_fp32
    energy_int8 = (mean_latency_int8 / 1000.0) * power_int8

    results = {
        "fp32_baseline": {
            "model_name": "MobileNetV2 (FP32 Live)",
            "latency_ms": round(mean_latency_fp32, 2),
            "latency_p90_ms": round(p90_latency_fp32, 2),
            "throughput_fps": round(throughput_fp32, 2),
            "ram_mb": round(ram_used_fp32, 2),
            "model_size_mb": round(fp32_size, 2),
            "avg_power_w": power_fp32,
            "energy_j": round(energy_fp32, 5),
            "energy_efficiency_ips_w": round(1.0 / energy_fp32 if energy_fp32 > 0 else 0, 2),
            "measurement_status": "live_host_cpu; power_assumed"
        },
        "int8_quantized": {
            "model_name": "MobileNetV2 (INT8 Live)",
            "latency_ms": round(mean_latency_int8, 2),
            "latency_p90_ms": round(p90_latency_int8, 2),
            "throughput_fps": round(throughput_int8, 2),
            "ram_mb": round(ram_used_int8, 2),
            "model_size_mb": round(int8_size, 2),
            "avg_power_w": power_int8,
            "energy_j": round(energy_int8, 5),
            "energy_efficiency_ips_w": round(1.0 / energy_int8 if energy_int8 > 0 else 0, 2),
            "measurement_status": "live_host_cpu; power_assumed"
        }
    }
    for key, path, samples in (
        ("fp32_baseline", fp32_path, latencies_fp32),
        ("int8_quantized", int8_path, latencies_int8),
    ):
        row = results[key]
        row.update({
            "latencies_ms": samples,
            "host": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "input_kind": "synthetic_uniform_minus1_plus1_seed42",
            "warmup_runs": 1,
            "timed_runs": num_runs,
            "memory_scope": "whole_process_RSS_MiB; models coexist; not an isolated comparison",
            "size_unit": "MiB",
            "avg_power_w": None,
            "energy_j": None,
            "energy_efficiency_ips_w": None,
            "measurement_status": "live_host_cpu; accuracy_and_power_not_measured",
        })
        with open(path, "rb") as model_file:
            row["model_sha256"] = hashlib.sha256(model_file.read()).hexdigest()
    return results

def print_comparison_table(results, mode="Emulated Raspberry Pi 4 Profile"):
    """Prints a beautiful, formatted comparison table."""
    print("\n" + "="*82)
    print(f" EDGE INFRASTRUCTURE BENCHMARKING REPORT ({mode.upper()}) ")
    print("="*82)
    print(f" {'Performance Metric':<30} | {'FP32 Baseline':<22} | {'INT8 Quantized TFLite':<22} ")
    print("-"*82)

    fp32 = results["fp32_baseline"]
    int8 = results["int8_quantized"]

    print(json.dumps(results, indent=2))
    print("Reference profiles are illustrative. Live timing uses synthetic inputs.")
    print("Live RSS includes coexisting models and cannot isolate model memory.")


def main():
    parser = argparse.ArgumentParser(description="Edge Deployment Benchmarking Script")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--force-emulation", action="store_true", help="Force emulation of Raspberry Pi 4 profile")
    parser.add_argument("--save", type=str, default="benchmark_run.json", help="Path to save results in JSON format")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.force_emulation:
        results = REFERENCE_PROFILE
        mode = "Explicit reference profile; no hardware measurements"
    else:
        if tf is None or np is None or psutil is None:
            parser.error("Live mode requires TensorFlow, NumPy and psutil; use --force-emulation for reference data.")
        results = run_actual_benchmark(num_runs=args.runs)
        mode = "Live host CPU diagnostic; synthetic input"

    print_comparison_table(results, mode=mode)

    # Save results to JSON
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Benchmark results successfully exported to {args.save}")

if __name__ == "__main__":
    main()
