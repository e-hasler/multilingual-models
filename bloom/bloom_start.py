
import os


os.environ["KERAS_BACKEND"] = "torch"  # Use PyTorch instead of JAX

script_dir = os.path.dirname(os.path.abspath(__file__))  # folder containing your script
output_path = os.path.join(script_dir, "bloom_outputs.txt")


import keras
import keras_hub

# When running only inference, bfloat16 saves memory usage significantly.
keras.config.set_floatx("bfloat16")

bloom_lm = keras_hub.models.BloomCausalLM.from_preset(
    "bloom_1.1b_multi"
)
bloom_lm.summary()

outputs = bloom_lm.generate([
    "Explain in simple terms differential equations.",
], max_length=512)

# Print outputs to console
for i, output in enumerate(outputs):
    print(f"Output {i+1}:\n{output}\n{'-'*40}\n")

# Save outputs to a text file
with open(output_path, "w") as f:
    for i, output in enumerate(outputs):
        f.write(f"Output {i+1}:\n{output}\n{'-'*40}\n")
print(f"Outputs saved to {output_path}")

# Clean up resources
keras.backend.clear_session()
del bloom_lm
