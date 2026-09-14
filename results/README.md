# Results

The checked-in JSON files are machine-readable sources for the manuscript:

- `windows_cpu_imagenette.json`: original conversion run, model provenance, marginal accuracy, and single-process latency.
- `paired_accuracy.json`: per-image predictions, disagreement counts, exact McNemar tests, and 100,000-resample paired bootstrap intervals.
- `per_tensor_accuracy.json`: full-validation accuracy for the per-tensor INT8 failure control.
- `repeated_latency.json`: five fresh-process replicates for every delegate, thread, batch, and granularity condition, including raw timings and host controls.
- `container_benchmark.json`: Docker versions, image size, five cold starts, 200 end-to-end request timings, resource limits, and RSS.

The repeated latency run used an AMD Ryzen 5 7600, TensorFlow Lite 2.15.1, Windows 24H2 build 26100.9278, the Balanced power plan, physical-core-spaced affinity, and above-normal benchmark priority. Docker Desktop was stopped, but interactive desktop applications remained open and sampled host load varied. Results on other systems are expected to differ.
