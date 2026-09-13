import tensorflow as tf
import numpy as np
import os

def representative_data_gen():
    """
    Generator for representative data used for INT8 quantization.
    The paper specifies that ~100-200 representative samples are typically used
    to calibrate the dynamic ranges of activations.
    """
    # In a real pipeline, you would load a subset of the actual dataset (e.g., ImageNet)
    for _ in range(100):
        # MobileNetV2 expects 224x224x3 input
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

def apply_int8_quantization(keras_model, output_path):
    """
    Implements INT8 Post-Training Quantization (PTQ) as detailed in Section III.A.
    This reduces model size by 4x and optimizes for edge CPU instructions (e.g., ARM NEON).
    """
    print(f"Starting INT8 Quantization for {output_path}...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model_quant = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model_quant)
    
    print(f"Quantization complete. Saved to: {output_path}")
    print(f"Model size reduced to: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # Example usage with MobileNetV2
    from model_factory import get_mobilenet_v2
    
    base_model = get_mobilenet_v2()
    os.makedirs("models", exist_ok=True)
    apply_int8_quantization(base_model, "models/mobilenet_v2_int8.tflite")
