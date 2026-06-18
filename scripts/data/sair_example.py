import os
import tarfile
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd


def load_sair_parquet_sample(destination_dir: str, n_rows: int = 5) -> pd.DataFrame:
    """Downloads sair.parquet and returns only the first n_rows rows."""
    repo_id = "SandboxAQ/SAIR"
    parquet_filename = "sair.parquet"

    os.makedirs(destination_dir, exist_ok=True)
    download_path = os.path.join(destination_dir, parquet_filename)

    print(f"Downloading '{parquet_filename}'...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=parquet_filename,
            repo_type="dataset",
            local_dir=destination_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return None

    try:
        df = pd.read_parquet(download_path)
        sample = df.head(n_rows)
        print(f"Loaded {len(sample)} rows (of {len(df)} total).")

        # Save a small sample parquet and delete the full one to save space
        sample_path = os.path.join(destination_dir, "sair_sample.parquet")
        sample.to_parquet(sample_path, index=False)
        os.remove(download_path)
        print(f"Saved sample to '{sample_path}', deleted full parquet.")
        return sample
    except Exception as e:
        print(f"Failed to load parquet: {e}")
        return None


def download_extract_few_structures(destination_dir: str, n_files: int = 3):
    """Downloads the first tar.gz from SAIR and extracts only n_files structures from it."""
    repo_id = "SandboxAQ/SAIR"
    repo_folder = "structures_compressed"

    os.makedirs(destination_dir, exist_ok=True)

    print("Listing repository files...")
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
        repo_tars = [
            f for f in all_files
            if f.startswith(repo_folder + "/") and f.endswith(".tar.gz")
        ]
    except Exception as e:
        print(f"Could not list repo files: {e}")
        return

    if not repo_tars:
        print("No tar.gz files found.")
        return

    # Just grab the first one
    repo_filepath = repo_tars[0]
    filename = repo_filepath.split("/")[-1]
    download_path = os.path.join(destination_dir, repo_folder, filename)

    print(f"Downloading '{filename}'...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_filepath,
            repo_type="dataset",
            local_dir=destination_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print(f"Extracting first {n_files} structures...")
    try:
        with tarfile.open(download_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in members[:n_files]:
                tar.extract(member, path=destination_dir)
                print(f"  Extracted: {member.name}")
    except Exception as e:
        print(f"Extraction failed: {e}")

    # Delete the tar to free space
    if os.path.exists(download_path):
        os.remove(download_path)
        print(f"Deleted '{download_path}' to save space.")

    print("Done.")


if __name__ == "__main__":
    output_dir = "./sair_example_data"

    print("=== Parquet sample ===")
    df = load_sair_parquet_sample(destination_dir=output_dir, n_rows=5)
    if df is not None:
        print(df)

    print("\n=== Structure sample ===")
    download_extract_few_structures(destination_dir=output_dir, n_files=3)
