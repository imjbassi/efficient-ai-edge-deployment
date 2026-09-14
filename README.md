# Efficient Packaging and Deployment of AI Models for Edge Inference

This repository contains an auditable reference implementation for packaging a MobileNetV2 image classifier for edge inference. It combines optional pruning, full-integer TensorFlow Lite conversion, a FastAPI service, and Docker Compose deployment.

## Included artifacts

- [Current compiled manuscript](output/pdf/main.pdf) (IEEE conference layout; reference results, pending empirical validation).
- `src/`: model construction, quantization, benchmarking, and serving code.
- `paper/`: the IEEEtran manuscript source and bibliography.
- `benchmark_results.json`: a reference/emulated Raspberry Pi 4 profile.
- `Dockerfile` and `docker-compose.yml`: container packaging and runtime limits.
- `Expanded_Efficient_Packaging_and_Deployment_of_AI_Models_for_Edge_Inference.pdf`: the original supplied manuscript artifact.

## Evidence status

The reference profile reports 13.6 MB to 3.3 MB model storage and 220 ms to 95 ms mean latency for FP32 and INT8 configurations. These values are retained for traceability, but the repository does not include raw hardware logs, power traces, validation predictions, or calibration images. Treat them as reference/emulated values, not as independently reproduced physical measurements. Energy values are derived from assumed power and latency.

The manuscript intentionally does not claim ImageNet accuracy, Jetson/Coral performance, batching gains, or container overhead until the corresponding artifacts are added.

## Quantization with real calibration data

The conversion scripts reject random calibration arrays. Provide a directory of representative `.jpg`, `.jpeg`, `.png`, or `.bmp` images and set:

```text
$env:EDGE_CALIBRATION_DIR = 'C:\path\to\calibration-images'
python src\quantize.py
```

The images are converted to RGB, resized to `224x224`, and normalized with MobileNetV2 preprocessing. The calibration set should be documented and hashed for a publishable experiment.

## Run the service

Place a valid model at `models/mobilenet_v2_int8.tflite`, install `requirements.txt`, and start:

```text
uvicorn src.deploy:app --host 0.0.0.0 --port 8000
```

The service exposes `/health`, `/metadata`, and `/predict`. Missing models return HTTP 503 from health and prediction endpoints. Predictions return class indices and scores; labels are null because no verified class mapping is bundled. Set `EDGE_ALLOW_MOCK=1` only for demonstrations that explicitly need the legacy simulated fallback. Model updates require a service restart; a bind mount alone does not reload the interpreter.

## Run the benchmark

```text
python src/benchmark.py --force-emulation --save reference_run.json
```

Live mode requires TensorFlow, NumPy and psutil and fails on errors. It retains per-run timings, model hashes, host and runtime metadata, and leaves power/energy unmeasured. It uses seeded synthetic inputs, one warm-up per backend, and coexisting models in one process: RSS cannot be used to claim isolated memory savings. It compares Keras FP32 against TFLite INT8, confounding precision and backend. Controlled hardware and labeled accuracy experiments remain necessary.

## Build the container

```text
docker compose up --build
```

The model directory is mounted at `/app/models`; the Compose CPU and memory limits are deployment policy, not evidence of measured performance.

## Compile the paper

The checked-in PDF was compiled with Tectonic 0.17.0 and visually inspected. From the repository root, run `tectonic paper/main.tex --outdir output/pdf`. Alternatively, from `paper/`, run `pdflatex main.tex`, `bibtex main`, and `pdflatex main.tex` twice. The root-level original PDF is preserved as a historical draft; it is not the revised manuscript.

See [publication status](PUBLICATION_STATUS.md) for validation and outstanding submission requirements. Model conversion requires a separate TensorFlow environment; `requirements.txt` covers Linux edge serving with a compatible TFLite wheel, not training or every host platform.
