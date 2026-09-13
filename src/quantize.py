import tensorflow as tf
import numpy as np

def representative_data_gen():
    """
    Generator for representative data used for INT8 quantization.
    In a real scenario, this would use a subset of the training/validation data.
    """
    for _ in range(100):
        # Generate dummy data matching MobileNetV2 input shape (224, 224, 3)
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

def quantize_model(keras_model_path, output_tflite_path):
    """
    Applies INT8 Post-Training Quantization as described in the paper.
    Reduces model size by ~4x and speeds up inference on edge CPUs.
    """
    print(f"Loading model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("Converting to INT8 TFLite...")
    tflite_model = converter.convert()
    
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Quantized model saved to {output_tflite_path}")

if __name__ == "__main__":
    import os
    from model_factory import get_mobilenet_v2
    
    # Save a baseline model first
    model = get_mobilenet_v2()
    model_path = "mobilenet_v2.h5"
    model.save(model_path)
    
    # Quantize
    quantize_model(model_path, "src/mobilenet_v2_quant.tflite")
