from argon.datasets.envs.pusht import load_pytorch_pusht_data
import jax
import jax.numpy as jnp
import jax.tree_util as tree

# Path to your .zarr.zip data
pusht_data_path = "/home/qul12/claireji/mode-collapse/argon/pusht_data.zarr"

# Load using pytorch-compatible function that doesn’t require `graphdef`
print("Loading PushT data...")
dataset = load_pytorch_pusht_data(pusht_data_path)
print(f"Loaded {len(dataset.sequences)} trajectories.")