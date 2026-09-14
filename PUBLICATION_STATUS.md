# Publication status

The repository is ready for public release as a reproducible research artifact and preprint. The manuscript has an explicit claim, paired accuracy statistics, independent-process latency uncertainty, backend and quantization controls, measured packaging outcomes, a publication figure, machine-readable raw observations, model hashes, tests, pinned environments, citation metadata, and a hardened container.

## Evidence included

- Full 3,925-image paired Imagenette evaluation with exact McNemar tests and 100,000-resample paired bootstrap intervals.
- Per-channel and per-tensor INT8 controls generated from the same graph and 200-image calibration set.
- Twenty fresh-process latency replicates per configuration across backends, 1/2/4/6 threads, and batch 1/4, with raw timings and host controls.
- Prediction-distribution and depthwise-channel-range evidence for the per-tensor numerical collapse.
- Five container cold starts, 200 localhost HTTP requests, image size, RSS, runtime identity, and enforced resource limits.
- Full dataset digest, reproducible TFLite binaries, figure-generation code, manuscript source, compiled PDF, and continuous integration.

## Scope and limitations

The claims apply to the reported MobileNetV2, Imagenette, Windows/x86 host, TensorFlow Lite 2.15.1 runtime, and Docker Desktop environment. The paper does not claim full-ImageNet generality, ARM or accelerator performance, energy or thermal behavior, concurrent serving throughput, or multi-model generality. Interactive applications deliberately remained open during host timing to represent a developer-workstation deployment host. Twenty process-level replicates and paired intervals expose the resulting host variation; this is not an idle-lab benchmark.

## Remaining submission choices

The venue-independent engineering and measurement work is complete except for packaging measurements if Docker Desktop is unavailable. Peer-reviewed submission still requires the author's venue choice, venue template and page limit, anonymization policy, author-affiliation and funding confirmation, and disclosure or artifact forms. A Zenodo DOI requires linking the final GitHub release from the author's Zenodo account. Broader empirical claims require new experiments on full ImageNet, additional architectures, and named edge devices.
