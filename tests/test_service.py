import asyncio
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from fastapi import HTTPException, UploadFile
from src import deploy


def png(value):
    data = io.BytesIO()
    Image.new('RGB', (2, 2), (value, value, value)).save(data, format='PNG')
    return data.getvalue()


class ServiceTests(unittest.TestCase):
    def test_integer_preprocessing_normalizes_before_quantizing(self):
        for dtype, zero in [(np.int8, 0), (np.uint8, 128)]:
            with self.subTest(dtype=dtype), patch.object(deploy, 'input_details', [{'dtype': dtype, 'quantization': (1/128, zero)}]):
                for value in (0, 128, 255):
                    expected = np.clip(round((value/127.5-1)*128+zero), np.iinfo(dtype).min, np.iinfo(dtype).max)
                    actual = deploy.preprocess_image(png(value))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(np.all(actual == expected))

    def test_missing_model_fails_prediction_and_health(self):
        with patch.object(deploy, 'interpreter', None), patch.object(deploy, 'ALLOW_MOCK', False):
            for operation in (deploy.health(), deploy.predict(UploadFile(file=io.BytesIO(png(0))))):
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(operation)
                self.assertEqual(error.exception.status_code, 503)

    def test_invalid_image_returns_bad_request(self):
        with self.assertRaises(HTTPException) as error:
            deploy.preprocess_image(b'not an image')
        self.assertEqual(error.exception.status_code, 400)

    def test_output_dequantizes_once_and_does_not_invent_labels(self):
        class Interpreter:
            def set_tensor(self, index, data): pass
            def invoke(self): pass
            def get_tensor(self, index): return np.array([[-128, 0, 64, 100]], dtype=np.int8)
        inputs = [{'index': 0, 'dtype': np.float32}]
        outputs = [{'index': 1, 'dtype': np.int8, 'quantization': (1/256, -128)}]
        with patch.object(deploy, 'interpreter', Interpreter()), patch.object(deploy, 'input_details', inputs), patch.object(deploy, 'output_details', outputs):
            result = asyncio.run(deploy.predict(UploadFile(file=io.BytesIO(png(128)))))
        first = result['predictions'][0]
        self.assertEqual(first['class_idx'], 3)
        self.assertAlmostEqual(first['confidence'], 228/256, places=3)
        self.assertIsNone(first['label'])
        self.assertGreaterEqual(result['latency_metadata']['postprocess_time_ms'], 0)


if __name__ == '__main__':
    unittest.main()
