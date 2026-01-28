
import os


os.environ["KERAS_BACKEND"] = "torch"  # Use PyTorch instead of JAX

import keras
import keras_hub

# When running only inference, bfloat16 saves memory usage significantly.
keras.config.set_floatx("bfloat16")

bloom_lm = keras_hub.models.BloomCausalLM.from_preset(
    "bloom_560m_multi"
)
bloom_lm.summary()

outputs = bloom_lm.generate([
    "Explain in simple terms differential equations.",
], max_length=512)

# Save outputs to a text file
with open("bloom_outputs.txt", "w") as f:
    for i, output in enumerate(outputs):
        f.write(f"Output {i+1}:\n{output}\n{'-'*40}\n")
print("Outputs saved to bloom_outputs.txt")

# Clean up resources
keras.backend.clear_session()
del bloom_lm
