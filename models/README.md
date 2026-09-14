# Models

Both files were exported from the same TensorFlow 2.15.1 MobileNetV2 graph with ImageNet weights. The INT8 artifact used 200 checksum-verified Imagenette training images for representative calibration.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mobilenet_v2_fp32.tflite` | 13,986,764 | `1029ab065d2f8a7225f81b1204f5df9cc947ab6756a0b5f77dd4350dcc2a35b7` |
| `mobilenet_v2_int8.tflite` | 3,991,688 | `ea1a077a43691c77c224d473cb30ca637733adc219df994e74973b98e0b66167` |

Regenerate them with `src/experiment.py` rather than editing or replacing them manually. Pretrained weights remain subject to their upstream terms.
