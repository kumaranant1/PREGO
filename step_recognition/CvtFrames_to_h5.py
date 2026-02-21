# @title frames to HDF5
import h5py
import os
import os.path as osp
import numpy as np
import json
import yaml
from tqdm.notebook import tqdm  # Use notebook version for better progress bars

# --- CONFIGURATION ---
config_path = "./configs/miniroad_assembly101-O.yaml"

# 1. Load Config
print(f"Loading configuration from {config_path}...")
with open(config_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

root_path = cfg['root_path']
data_name = cfg['data_name']
rgb_type = cfg['rgb_type']
flow_type = cfg['flow_type']
video_list_path = cfg['video_list_path']

# Output path
output_file = f"/{data_name}_features.h5"

# Optimization: Check if we should skip flow storage
# Your code zeroes out flow for 'flow_anet_resnet50', so we won't save it to disk.
is_zero_flow = (flow_type == 'flow_anet_resnet50')

print(f"Dataset: {data_name}")
print(f"Root Path: {root_path}")
print(f"Feature Types -> RGB: {rgb_type}, Flow: {flow_type}")
if is_zero_flow:
    print("Optimization: 'flow_anet_resnet50' detected. Skipping flow storage to save space.")

# 2. Load Video List
if not osp.exists(video_list_path):
    raise FileNotFoundError(f"Video list not found at: {video_list_path}")

with open(video_list_path, 'r') as f:
    data_dict = json.load(f)
    video_lists = data_dict[data_name]

# Combine train and test sets to pack everything in one file
all_vids = list(set(video_lists.get('train_session_set', []) + video_lists.get('test_session_set', [])))
print(f"Found {len(all_vids)} unique videos to pack.")

# 3. Create HDF5 File
print(f"Creating HDF5 file at: {output_file}")

# Open HDF5 file
# 'w' mode will overwrite if the file already exists
with h5py.File(output_file, 'w') as hf:
    success_count = 0
    skip_count = 0

    # Iterate with progress bar
    for vid in tqdm(all_vids, desc="Packing Videos"):
        try:
            # --- Process RGB ---
            rgb_path = osp.join(root_path, rgb_type, vid + ".npy")

            if osp.exists(rgb_path):
                rgb_data = np.load(rgb_path)

                # Create a group for the video
                grp = hf.create_group(vid)

                # Store RGB data (compressed to save space)
                grp.create_dataset("rgb", data=rgb_data, dtype='float32', compression="gzip")

                # --- Process Flow ---
                if not is_zero_flow:
                    flow_path = osp.join(root_path, flow_type, vid + ".npy")
                    if osp.exists(flow_path):
                        flow_data = np.load(flow_path)
                        grp.create_dataset("flow", data=flow_data, dtype='float32', compression="gzip")

                success_count += 1
            else:
                skip_count += 1

        except Exception as e:
            print(f"Error processing {vid}: {e}")
            skip_count += 1

print("\n" + "="*30)
print("PACKING COMPLETE")
print(f"Successfully packed: {success_count} videos")
print(f"Skipped/Missing: {skip_count} videos")
print(f"Output saved to: {output_file}")
print("="*30)