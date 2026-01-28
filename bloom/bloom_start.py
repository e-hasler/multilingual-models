
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

print("outputs type:", type(outputs))
print("Raw outputs:", outputs)
print("\nFormatted outputs:")
for i, output in enumerate(outputs):
    print(f"Output {i+1}:\n{output}\n{'-'*40}")