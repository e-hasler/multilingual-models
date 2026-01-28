
import os
import pandas as pd
from IPython.display import display

os.environ["KERAS_BACKEND"] = "torch"  # Use PyTorch instead of JAX

script_dir = os.path.dirname(os.path.abspath(__file__))  # folder containing your script
output_path = os.path.join(script_dir, "bloom_outputs.tsv")
    # The output file extension is already set to .tsv


import keras
import keras_hub

# When running only inference, bfloat16 saves memory usage significantly.
keras.config.set_floatx("bfloat16")


models = ["bloom_560m_multi", "bloom_1.1b_multi", "bloom_1.7b_multi", "bloom_3b_multi",
          "bloomz_560m_multi", "bloomz_1.1b_multi", "bloomz_1.7b_multi", "bloomz_3b_multi"]
queries = [
    "Explain in simple terms differential equations.",
]

# Create dataframe with outputs

df = pd.DataFrame({"model": models, "output": [""] * len(models)})
# Duplicate each row per query
df = df.loc[df.index.repeat(len(queries))].reset_index(drop=True)
df["query"] = queries * len(models)

display(df.head())

# Load models from KerasHub and generate outputs
for model in models[:0]: #TODO: change to models to run all models
    for query in queries:
        # Load model
        bloom_lm = keras_hub.models.BloomCausalLM.from_preset(
            model
        )
        bloom_lm.summary()

        # Generate output
        outputs = bloom_lm.generate([
            query,
        ], max_length=512)

        df.loc[(df["model"] == model) & (df["query"] == query), "output"] = outputs[0]
        print(f"Generated output for model {model} is {outputs[0]}")



# Save dataframe as TSV for better readability
df.to_csv(output_path, sep="\t", index=False)
print(f"Outputs saved as TSV in {output_path}")

# Clean up resources
keras.backend.clear_session()
del bloom_lm
