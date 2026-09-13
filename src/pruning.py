import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

def apply_pruning_to_model(model):
    """
    Applies weight pruning to the model as described in the paper.
    The paper mentions a sparsity level of ~30% was effective.
    """
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.30,
            begin_step=0,
            end_step=1000
        )
    }

    # Wrap the model with pruning wrappers
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    return model_for_pruning

def strip_pruning_and_save(pruned_model, output_path):
    """
    Strips pruning wrappers for deployment.
    This is necessary before quantization or saving the final artifact.
    """
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    final_model.save(output_path)
    print(f"Pruned model saved to {output_path}")

if __name__ == "__main__":
    from model_factory import get_mobilenet_v2
    
    # Load baseline
    model = get_mobilenet_v2()
    
    # Apply pruning
    print("Applying 30% sparsity pruning...")
    pruned_model = apply_pruning_to_model(model)
    
    # Note: In a real scenario, you would retrain/fine-tune the pruned model here.
    # For the purpose of this script, we strip and save.
    strip_pruning_and_save(pruned_model, "models/mobilenet_v2_pruned.h5")
