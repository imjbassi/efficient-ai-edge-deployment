# Models

All files were exported from the same TensorFlow 2.15.1 MobileNetV2 graph with ImageNet weights. The INT8 artifacts used the same 200 checksum-verified Imagenette training images for representative calibration.

| Artifact | Quantization | Bytes | SHA-256 |
|---|---|---:|---|
| `mobilenet_v2_fp32.tflite` | FP32 | 13,986,764 | `1029ab065d2f8a7225f81b1204f5df9cc947ab6756a0b5f77dd4350dcc2a35b7` |
| `mobilenet_v2_int8.tflite` | Full INT8, per-channel weights | 3,991,688 | `ea1a077a43691c77c224d473cb30ca637733adc219df994e74973b98e0b66167` |
| `mobilenet_v2_int8_per_tensor.tflite` | Full INT8, per-tensor control | 3,583,560 | `38ca72c6abce6ad9cef394478a1cb13308e731564ab0ddd8c96591f748e2c7cb` |

The per-tensor file is an experimental failure control: it reaches only 1.04% top-1 accuracy on Imagenette and must not be deployed. The service intentionally uses the per-channel model.

Regenerate the primary artifacts with `src/experiment.py` and the control with `scripts/create_per_tensor_model.py`; do not edit binaries manually. Pretrained weights remain subject to their upstream terms.
