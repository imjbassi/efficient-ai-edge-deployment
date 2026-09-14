# Efficient AI Edge Deployment

[![CI](https://github.com/imjbassi/efficient-ai-edge-deployment/actions/workflows/ci.yml/badge.svg)](https://github.com/imjbassi/efficient-ai-edge-deployment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--3633--3220-A6CE39.svg)](https://orcid.org/0009-0006-3633-3220)

A reproducible MobileNetV2 case study covering FP32 and full-integer TensorFlow Lite conversion, validation on Imagenette, single-thread CPU benchmarking, and deployment through FastAPI and Docker.

## Measured result

The checked-in experiment evaluated every image in the 3,925-image Imagenette validation split. Calibration used 200 deterministically selected training images. Latency used 500 validation inputs after 50 warm-up runs on an AMD Ryzen 5 7600, TensorFlow Lite 2.15.1, XNNPACK, and one interpreter thread; preprocessing was excluded.

| Metric | FP32 | INT8 |
|---|---:|---:|
| Model size | 13.34 MiB | 3.81 MiB |
| Top-1 accuracy | 66.73% | 65.81% |
| Top-5 accuracy | 90.75% | 89.81% |
| Mean latency | 8.53 ms | 6.46 ms |
| 90th-percentile latency | 9.88 ms | 7.29 ms |

INT8 reduced storage by 71.5%, improved mean host-CPU latency by 1.32x, and decreased top-1 accuracy by 0.92 percentage points. These are host measurements, not Raspberry Pi, Jetson, accelerator, power, or container-overhead results.

- [Compiled paper](output/pdf/main.pdf)
- [Machine-readable results](results/windows_cpu_imagenette.json)
- [Publication status](PUBLICATION_STATUS.md)
- [Citation metadata](CITATION.cff)

## Reproduce the experiment

Python 3.10 is required. The download script pins and verifies the Imagenette archive SHA-256 digest before extraction.

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

Use `--reuse-models` to repeat evaluation and timing without reconverting the checked-in models. Exact results vary by CPU and system load. The result JSON retains raw timings, model and dataset hashes, environment metadata, accuracy counts, and Wilson confidence intervals.

## Run the inference service

On Linux, install the lightweight serving dependencies and start the service:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn src.deploy:app --host 0.0.0.0 --port 8000
```

Then submit an image:

```bash
curl -F "file=@image.jpg" http://localhost:8000/predict
```

The API exposes `/health`, `/metadata`, and `/predict`. It validates image data and upload size, uses the paper's resize/crop/scale pipeline, reads quantization parameters from the model, and serializes interpreter access. Prediction output contains ImageNet class indices and scores; labels are intentionally omitted rather than guessed.

## Run with Docker

```bash
docker compose up --build
```

The image includes the checked-in INT8 model, runs as a non-root user, uses a read-only filesystem, and defines CPU, memory, and upload-size limits. These limits are deployment policy rather than benchmark evidence.

## Build and test

```bash
python -m compileall -q src tests scripts
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
tectonic paper/main.tex --outdir output/pdf
```

The experiment environment installs full TensorFlow and is intentionally separate from the smaller Linux serving environment. The MIT license applies to repository code and documentation; pretrained model weights and Imagenette remain subject to their upstream terms.

## Repository layout

```text
models/       Reproducible FP32 and INT8 TFLite artifacts
output/pdf/   Compiled manuscript
paper/        LaTeX source and bibliography
results/      Raw measurements and experiment metadata
scripts/      Checksum-verified dataset acquisition
src/          Experiment, benchmark entry point, and service
tests/        Service regression tests
```
