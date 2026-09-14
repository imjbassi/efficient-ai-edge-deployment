# Publication status

The repository is ready for public release as a reproducible research artifact and preprint. The manuscript has an explicit claim, paired accuracy statistics, independent-process latency uncertainty, backend and quantization controls, measured packaging outcomes, a publication figure, machine-readable raw observations, model hashes, tests, pinned environments, citation metadata, and a hardened container.

## Evidence included

- Full 3,925-image paired Imagenette evaluation with exact McNemar tests and 100,000-resample paired bootstrap intervals.
- Per-channel and per-tensor INT8 controls generated from the same graph and 200-image calibration set.
- Five fresh-process latency replicates per configuration across delegates, 1/2/4/6 threads, and batch 1/4, with raw timings and host controls.
- Five container cold starts, 200 localhost HTTP requests, image size, RSS, runtime identity, and enforced resource limits.
- Full dataset digest, reproducible TFLite binaries, figure-generation code, manuscript source, compiled PDF, and continuous integration.

## Scope and limitations

The claims apply to the reported MobileNetV2, Imagenette, Windows/x86 host, TensorFlow Lite 2.15.1 runtime, and Docker Desktop environment. The paper does not claim full-ImageNet generality, ARM or accelerator performance, energy or thermal behavior, concurrent serving throughput, or multi-model generality. Interactive applications remained open during host timing, and the five process-level replicates leave wide intervals in some thread conditions.

## Remaining submission choices

No engineering blocker remains for a public repository or preprint. Peer-reviewed submission still requires the author's venue choice, venue template and page limit, anonymization policy, author-affiliation confirmation, and disclosure or artifact forms. Broader empirical claims require new experiments on full ImageNet, additional architectures, and named edge devices.
