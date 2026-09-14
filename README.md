# Efficient AI Edge Deployment

[![CI](https://github.com/imjbassi/efficient-ai-edge-deployment/actions/workflows/ci.yml/badge.svg)](https://github.com/imjbassi/efficient-ai-edge-deployment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--3633--3220-A6CE39.svg)](https://orcid.org/0009-0006-3633-3220)

Reproducible code and evidence for *When Does INT8 Actually Accelerate Edge Inference?* The project audits MobileNetV2 TensorFlow Lite conversion across quantization granularity, CPU delegates, thread counts, batch size, independent process restarts, and container serving.

## Main finding

INT8 was always smaller, but it was not intrinsically faster or accurate. On the complete 3,925-image Imagenette validation split, per-channel INT8 reduced model storage by 71.5% and changed top-1 accuracy by -0.92 percentage points (paired bootstrap 95% CI [-1.66, -0.18], exact McNemar p=.0165). On an AMD Ryzen 5 7600:

- One-thread XNNPACK produced a 1.43x paired geometric-mean speedup (95% CI [1.18, 1.73]) across five fresh processes.
- Built-in kernels made INT8 2.22x slower: FP32/INT8 speedup was 0.45x (95% CI [0.40, 0.51]).
- Per-tensor INT8 had similar latency to per-channel INT8 but collapsed to 1.04% top-1 accuracy.
- The production container was 339.58 MiB uncompressed, reached health in a median 1.69 s, used 77.36 MiB RSS after load, and delivered localhost HTTP p50/p99 latency of 46.34/71.85 ms.

These are measurements on the named Windows/x86 and Docker Desktop environment, not claims about ARM boards, accelerators, energy, or all neural networks.

- [Compiled paper](output/pdf/main.pdf)
- [Repeated-process latency results](results/repeated_latency.json)
- [Paired accuracy results](results/paired_accuracy.json)
- [Container benchmark results](results/container_benchmark.json)
- [Publication status](PUBLICATION_STATUS.md)
- [Citation metadata](CITATION.cff)

## Reproduce the experiments

Python 3.10 is required. The dataset downloader verifies the full Imagenette archive SHA-256 before extraction.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-experiment.txt
.venv\Scripts\python scripts\download_imagenette.py --destination data
$env:TF_ENABLE_ONEDNN_OPTS = "0"
.venv\Scripts\python src\experiment.py `
  --data-dir data\imagenette2-160 `
  --archive data\imagenette2-160.tgz `
  --output results\local_experiment.json `
  --calibration-samples 200 `
  --latency-samples 500 `
  --warmup 50 `
  --threads 1
```

Generate the per-tensor control, evaluate it, run the paired analysis, and execute the repeated-process matrix:

```powershell
.venv\Scripts\python scripts\create_per_tensor_model.py `
  --data-dir data\imagenette2-160
.venv\Scripts\python src\evaluate_model.py `
  --model models\mobilenet_v2_int8_per_tensor.tflite `
  --data-dir data\imagenette2-160 `
  --output results\per_tensor_accuracy.json
.venv\Scripts\python src\paired_accuracy.py `
  --data-dir data\imagenette2-160 `
  --output results\paired_accuracy.json
.venv\Scripts\python src\repeated_latency.py `
  --data-dir data\imagenette2-160 `
  --output results\repeated_latency.json
.venv\Scripts\python scripts\plot_latency_audit.py
```

The latency matrix is intentionally substantial. It starts a fresh process for every replicate and retains raw observations, initialization time, affinity, and sampled host load.

## Run and benchmark the service

```bash
docker compose up --build
```

Submit an image with `curl -F "file=@image.jpg" http://localhost:8000/predict`. The API exposes `/health`, `/metadata`, and `/predict`, validates image data and upload size, uses the paper's preprocessing, reads model quantization metadata, and serializes interpreter access.

Run the packaging benchmark from a separate terminal after Docker Desktop is ready:

```powershell
.venv\Scripts\python scripts\benchmark_container.py `
  --image data\imagenette2-160\val\n01440764\ILSVRC2012_val_00009111.JPEG `
  --output results\container_benchmark.json
```

The image contains only the serving dependencies and per-channel INT8 model. It runs as a non-root user with a read-only filesystem and explicit CPU, memory, and upload-size limits.

## Build and test

```powershell
python -m compileall -q src tests scripts
python -m unittest discover -s tests -v
python scripts\verify_artifacts.py
.venv\Scripts\python scripts\plot_latency_audit.py
tectonic paper/main.tex --outdir output/pdf
```

The experiment environment installs full TensorFlow and is separate from the smaller Linux serving environment. The MIT license applies to repository code and documentation; pretrained weights and Imagenette remain subject to their upstream terms.

## Repository layout

```text
models/       Hashed FP32, per-channel INT8, and per-tensor INT8 artifacts
output/pdf/   Compiled manuscript
paper/        IEEE-style LaTeX source, bibliography, and figure
results/      Raw paired, process-level, and container measurements
scripts/      Dataset, conversion, plotting, verification, and packaging tools
src/          Experiment, statistics, benchmark workers, and service
tests/        Service regression tests
```
