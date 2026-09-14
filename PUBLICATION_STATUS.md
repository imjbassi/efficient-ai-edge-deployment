# Publication status

The repository is ready to publish as a reproducible research artifact and preprint. The manuscript compiles, its headline values are backed by the checked-in machine-readable result, both evaluated model binaries are versioned with hashes, the serving path is tested, and the project has citation metadata, an MIT license, continuous integration, pinned environments, and a hardened container configuration.

## Evidence included

- Full 3,925-image Imagenette validation evaluation for FP32 and INT8 models.
- Deterministic 200-image calibration selection and checksum-verified dataset acquisition.
- Five hundred raw latency samples per model after 50 warm-ups.
- CPU, runtime, thread count, preprocessing, confidence intervals, and artifact hashes.
- Reproducible TFLite binaries and a service that uses matching preprocessing.
- Successful Linux/amd64 container build and smoke test covering health, metadata,
  real-image inference, non-root execution, resource limits, and read-only storage.

## Claims deliberately excluded

The paper does not claim Raspberry Pi, Jetson, Coral, ARM, accelerator, power, thermal, memory, HTTP, or container-overhead performance. It also does not present pruning results or treat Imagenette accuracy as a replacement for full ImageNet evaluation.

## Remaining submission choices

No engineering blocker remains for a public repository or preprint release. A formal venue submission still requires the author's choice of venue, its required template and page limit, anonymization if applicable, author-affiliation confirmation, and any venue-specific disclosure or artifact forms. Extending the claims to named edge devices requires running the checked-in model hashes and protocol on those devices.
