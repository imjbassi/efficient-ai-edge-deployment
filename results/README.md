# Results

The checked-in JSON files are machine-readable sources for the manuscript:

- `windows_cpu_imagenette.json`: original conversion run, model provenance, marginal accuracy, and single-process latency.
- `paired_accuracy.json`: per-image predictions, disagreement counts, exact McNemar tests, and 100,000-resample paired bootstrap intervals.
- `per_tensor_accuracy.json`: full-validation accuracy for the per-tensor INT8 failure control.
- `repeated_latency.json`: 20 fresh-process replicates for every backend, thread, batch, and granularity condition, including raw timings and host controls.
- `quantization_diagnostics.json`: per-tensor prediction distribution and per-channel depthwise-weight scale/range statistics.
- `container_benchmark.json`: Docker versions, image size, 20 cold starts, 200 end-to-end request timings, server-stage decomposition, resource limits, and RSS.
- `container_native_latency.json`: one direct 500-image invocation run inside the service image, with model and input-selection hashes plus delegated-op evidence.
- `image_size_comparison.json`: uncompressed and deterministic gzip-compressed image sizes for pinned slim and standard Python bases.

`scripts/benchmark_image_size.py` regenerates the image-size comparison from the pinned bases.

The repeated latency run used an AMD Ryzen 5 7600, TensorFlow Lite 2.15.1, Windows 24H2 build 26100.9278, the Balanced power plan, physical-core-spaced affinity, and above-normal benchmark priority. Docker Desktop was stopped, but interactive desktop applications remained open and sampled host load varied. Results on other systems are expected to differ.
