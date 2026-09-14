# Publication status

The revised PDF in `output/pdf/main.pdf` compiles in IEEE conference layout. It is an engineering reference manuscript with illustrative results, not a validated empirical study. The previous claim that it was fully publication-ready was too strong.

The original draft's author name, email, and Grand Canyon University affiliation are preserved. Confirm the affiliation and the target venue's submission format before submitting.

## Remaining work for an empirical submission

- Supply representative calibration images and a labeled evaluation set with access instructions and hashes; no ImageNet evaluation was run.
- Export and archive actual model artifacts; execute pruning training with callbacks if pruning results will be claimed.
- Collect raw measurements on the named edge devices, including repeated runs, software versions, thread settings and thermal conditions.
- Measure power with an identified meter and provide synchronized traces before reporting energy efficiency.
- Compare identical backends/models across native and container serving with matched inputs to isolate container overhead.
- Choose a venue and assess novelty: combining existing quantization, serving, and packaging tools alone does not establish a new compression method.
- Select a repository license before advertising unrestricted reuse; no license has been invented on the author's behalf.

## Validation scope

PDF compilation and visual review, Python syntax checks, explicit-emulation benchmark execution, failed-live-run behavior, and service regression tests are checked locally. Tests cover normalized integer inputs, unavailable models, invalid image handling, and output scores with a stub interpreter. These checks do not validate real TensorFlow conversion, actual edge latency, classification accuracy, or a Docker image on ARM hardware.

The compiled PDF and sources are committed together. Build dependencies, virtual environments, preview images, and temporary logs remain local under ignored paths.
