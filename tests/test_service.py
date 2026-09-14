import asyncio
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException, Request, UploadFile
from PIL import Image

from src import deploy


def png(value: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (value, value, value)).save(output, format="PNG")
    return output.getvalue()


class FakeInterpreter:
    def set_tensor(self, index, value):
        self.input = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        return np.array([[-128, 0, 64, 100]], dtype=np.int8)


class ServiceTests(unittest.TestCase):
    def test_integer_preprocessing_uses_model_scale(self):
        detail = {"dtype": np.int8, "shape": [1, 224, 224, 3], "quantization": (1 / 128, 0)}
        actual = deploy.preprocess_image(png(0), detail)
        self.assertEqual(actual.dtype, np.int8)
        self.assertTrue(np.all(actual == -128))

    def test_invalid_image_is_rejected(self):
        detail = {"dtype": np.float32, "shape": [1, 224, 224, 3], "quantization": (0.0, 0)}
        with self.assertRaises(HTTPException) as error:
            deploy.preprocess_image(b"invalid", detail)
        self.assertEqual(error.exception.status_code, 400)

    def test_oversized_image_dimensions_are_rejected(self):
        detail = {"dtype": np.float32, "shape": [1, 224, 224, 3], "quantization": (0.0, 0)}
        with patch.object(deploy, "MAX_IMAGE_PIXELS", 1), self.assertRaises(HTTPException) as error:
            deploy.preprocess_image(png(0), detail)
        self.assertEqual(error.exception.status_code, 413)

    def test_missing_model_is_unhealthy(self):
        with patch.object(deploy.runtime, "interpreter", None), patch.object(deploy.runtime, "error", "missing"):
            with self.assertRaises(HTTPException) as error:
                deploy.health()
        self.assertEqual(error.exception.status_code, 503)

    def test_prediction_dequantizes_once(self):
        input_detail = {"index": 0, "dtype": np.float32, "shape": [1, 224, 224, 3], "quantization": (0.0, 0)}
        output_detail = {"index": 1, "dtype": np.int8, "shape": [1, 4], "quantization": (1 / 256, -128)}
        with (
            patch.object(deploy.runtime, "interpreter", FakeInterpreter()),
            patch.object(deploy.runtime, "input_detail", input_detail),
            patch.object(deploy.runtime, "output_detail", output_detail),
        ):
            scope = {"type": "http", "method": "POST", "path": "/predict", "headers": []}
            request = Request(scope)
            result = asyncio.run(
                deploy.predict(request, UploadFile(file=io.BytesIO(png(128))))
            )
            result = json.loads(result.body)
        self.assertEqual(result["predictions"][0]["class_index"], 3)
        self.assertAlmostEqual(result["predictions"][0]["score"], 228 / 256, places=5)
        self.assertIn("invoke", result["timing_ms"])


if __name__ == "__main__":
    unittest.main()
