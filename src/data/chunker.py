import os
import numpy as np
from glob import glob
import json

def chunk_mocap_file(file_path, window_size=30, step_size=15):
    """
    Load a .npy mocap file and split it into overlapping chunks.
    Returns a list of chunks and metadata for each chunk.
    """
    data = np.load(file_path).transpose(1, 0, 2)  # FIXED: ensure correct shape
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    num_timesteps = data.shape[0]

    chunks = []
    metadata = []

    chunk_id = 0
    for start in range(0, num_timesteps - window_size + 1, step_size):
        end = start + window_size
        chunk = data[start:end]  # shape: (window_size, joints, dims)
        chunks.append(chunk)

        metadata.append({
            "id": f"{file_name}_{start}_{end}",
            "file": file_name,
            "start": start,
            "end": end,
            "chunk_id": chunk_id
        })

        chunk_id += 1

    return np.stack(chunks), metadata



def chunk_all_files(
    input_dir="data/raw",
    output_dir="data/processed",
    metadata_path="data/processed/chunk_metadata.json",
    window_size=60,
    step_size=50
):
    """
    Process all .npy files in a directory and save the chunked sequences.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_metadata = []
    raw_files = sorted(glob(os.path.join(input_dir, "*.npy")))

    for file_path in raw_files:
        print(f"Chunking {file_path}...")
        chunks, metadata = chunk_mocap_file(file_path, window_size, step_size)

        # Save the full chunk array for this file
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, f"{file_name}_chunks.npy")
        np.save(out_path, chunks)

        all_metadata.extend(metadata)

    # Save metadata for all chunks across all files
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"\nSaved chunked data to: {output_dir}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    chunk_all_files()
