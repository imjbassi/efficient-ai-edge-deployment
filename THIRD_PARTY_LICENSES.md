# Third-party model and dataset notices

The repository's original code, manuscript source, and documentation are licensed under the repository [MIT License](LICENSE). That license does not relicense third-party software, pretrained parameters, or datasets.

## TensorFlow, Keras, and TensorFlow Lite

The conversion and runtime code uses TensorFlow/Keras 2.15.1 and TensorFlow Lite, distributed upstream under [Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE). The checked-in `.tflite` files are converted derivatives of Keras MobileNetV2 weights pretrained on ImageNet. The upstream weight download does not provide separate license metadata in this artifact, so the repository does not assert a new license for those parameters. Users should review the TensorFlow/Keras and ImageNet terms before redistribution or commercial use.

## Imagenette and ImageNet

[Imagenette](https://github.com/fastai/imagenette) is Jeremy Howard's ten-class subset of ImageNet. This repository does not redistribute the image archive: `scripts/download_imagenette.py` downloads it from the official fastai distribution and verifies its SHA-256. ImageNet states that access is for non-commercial research and educational use under its [access agreement](https://image-net.org/accessagreement); the repository's MIT license does not cover the images.

## Reproducibility boundary

Machine-readable predictions and aggregate measurements are checked in. The source dataset itself is excluded. Model filenames, hashes, conversion settings, framework versions, and upstream provenance are recorded so downstream users can make their own licensing determination.
