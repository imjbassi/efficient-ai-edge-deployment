# Results

`windows_cpu_imagenette.json` is the source of truth for the values reported in the manuscript and root README. It contains the complete environment record, model and dataset hashes, accuracy counts and confidence intervals, per-class counts, summary latency statistics, and every raw latency sample.

The latency run used one TensorFlow Lite XNNPACK thread on an otherwise idle AMD Ryzen 5 7600. Preprocessing was performed before timing. Results on other systems are expected to differ.
