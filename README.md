# Efficient AI Edge Deployment

[![CI](https://github.com/imjbassi/efficient-ai-edge-deployment/actions/workflows/ci.yml/badge.svg)](https://github.com/imjbassi/efficient-ai-edge-deployment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--3633--3220-A6CE39.svg)](https://orcid.org/0009-0006-3633-3220)

Reproducible code and evidence for *When Does INT8 Actually Accelerate Host-CPU Inference?* The project audits MobileNetV2 TensorFlow Lite conversion across quantization granularity, CPU backends, thread counts, batch size, independent process restarts, and container serving.

## Main finding

INT8 was always smaller, but it was not intrinsically faster or accurate. On the complete 3,925-image Imagenette validation split, per-channel INT8 reduced model storage by 71.5% and changed top-1 accuracy by -0.92 percentage points (paired bootstrap 95% CI [-1.66, -0.18], exact McNemar p=.0165). On an AMD Ryzen 5 7600:

- One-thread XNNPACK produced a 1.27x paired geometric-mean speedup (95% CI [1.20, 1.35]) across 20 fresh processes.
- TensorFlow Lite 2.15.1's `BUILTIN_WITHOUT_DEFAULT_DELEGATES` counterfactual made INT8 2.24x slower: FP32/INT8 speedup was 0.45x (95% CI [0.42, 0.47]). XNNPACK is the default CPU delegate.
- Per-tensor INT8 had similar latency to per-channel INT8 but collapsed to 1.04% top-1 accuracy. It used only 187 output classes, concentrated 19.6% of predictions in one class, and removed channel-specific scales from depthwise tensors whose per-channel range ratios reach 754x.
- The slim production image was 339.58 MiB uncompressed (89.2x its model) and 81.43 MiB as a compressed archive; the standard-base control was 1,543.65/393.44 MiB.
- Across 20 cold starts, median readiness was 1.52 s. Across 200 requests, HTTP p50/p99 was 49.87/73.96 ms and post-run RSS was 77.16 MiB.
- Mean HTTP latency decomposed into 38.66 ms container invocation, 10.96 ms client/loopback transport, and 2.35 ms parsing, upload, decode, resize, normalization, postprocessing, serialization, and residual framework work.
- A matched 500-image, one-thread run executed directly inside the container averaged 38.83 ms (p50 37.54 ms), 0.17 ms above the service's invocation mean (sampling noise). The 31.40 ms host-native-to-container-direct gap is therefore runtime/backend plus virtualization, not service wrapping; the INT8 graph had zero delegated operations, while an FP32 probe in the same runtime had one XNNPACK-delegated partition.

These are measurements on the named Windows/x86 and Docker Desktop environment, not claims about ARM boards, accelerators, energy, or all neural networks.

- [Compiled paper](output/pdf/main.pdf)
- [Repeated-process latency results](results/repeated_latency.json)
- [Paired accuracy results](results/paired_accuracy.json)
- [Container benchmark results](results/container_benchmark.json)
- [Container-native latency results](results/container_native_latency.json)
- [Container image-size comparison](results/image_size_comparison.json)
- [Quantization diagnostics](results/quantization_diagnostics.json)
- [Publication status](PUBLICATION_STATUS.md)
- [Citation metadata](CITATION.cff)
- [Windows Docker stale-socket recovery](docs/docker-desktop-windows-recovery.md)

## Reproduce the paper

Python 3.10 is required. The dataset downloader verifies the full Imagenette archive SHA-256 before extraction. After environment setup, one entry point validates the checked-in evidence, regenerates the figure, and compiles the manuscript:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-experiment.txt
.venv\Scripts\python scripts\download_imagenette.py --destination data
.venv\Scripts\python scripts\reproduce_paper.py --tectonic path\to\tectonic.exe
```

Rerun every native measurement (an overnight-scale job) with:

```powershell
.venv\Scripts\python scripts\reproduce_paper.py `
  --measure `
  --data-dir data\imagenette2-160 `
  --tectonic path\to\tectonic.exe
```

The latency matrix starts a fresh process for each of 20 replicates per condition and retains raw observations, initialization time, affinity, and sampled host load. A resumable checkpoint is updated after every process and removed only after successful completion.

## Run and benchmark the service

```bash
docker compose up --build
```

Submit an image with `curl -F "file=@image.jpg" http://localhost:8000/predict`. The API exposes `/health`, `/metadata`, and `/predict`, validates image data and upload size, uses the paper's preprocessing, reads model quantization metadata, and serializes interpreter access.

Run the packaging benchmark from a separate terminal after Docker Desktop is ready:

```powershell
.venv\Scripts\python scripts\benchmark_container.py `
  --image data\imagenette2-160\val\n01440764\ILSVRC2012_val_00009111.JPEG `
  --data-dir data\imagenette2-160 `
  --output results\container_benchmark.json
.venv\Scripts\python scripts\benchmark_image_size.py
```

Equivalently, run both packaging measurements and rebuild the PDF through the single entry point:

```powershell
.venv\Scripts\python scripts\reproduce_paper.py `
  --container `
  --image data\imagenette2-160\val\n01440764\ILSVRC2012_val_00009111.JPEG `
  --tectonic path\to\tectonic.exe
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

The experiment environment installs full TensorFlow and is separate from the smaller Linux serving environment. The MIT license applies to repository code and documentation; pretrained weights and Imagenette remain subject to their upstream terms. See [third-party notices](THIRD_PARTY_LICENSES.md) for the model and dataset provenance boundary.

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
