import gdown
import zipfile
import os
import pandas as pd
from pathlib import Path

def download_and_extract_kaggle_data(
    url: str = 'https://drive.google.com/uc?id=1QDHZrqQ8gbNW18Flxs5RCsC_tv7hxfeH',
    output_path: str = 'data/raw/sentiment_kaggle.zip',
    extract_dir: str = 'data/raw/sentiment_kaggle',
    parquet_output: str = 'data/raw/sentiment_kaggle.parquet'
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"⬇️ Downloading Kaggle sentiment140 dataset from Google Drive...")
    gdown.download(url, output_path, quiet=False)

    print(f"📦 Extracting to: {extract_dir}")
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(output_path)
    print("✅ Download and extraction complete.")

    # Find CSV file in the extracted directory
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("❌ No CSV file found in the extracted directory.")

    csv_path = os.path.join(extract_dir, csv_files[0])
    print(f"📄 Found CSV file: {csv_path}")

    print("📊 Loading CSV into DataFrame...")
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', header=None)

    # Save to Parquet
    Path(os.path.dirname(parquet_output)).mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_output, index=False)
    print(f"✅ Saved as Parquet to: {parquet_output}")

    # Remove the original CSV file
    os.remove(csv_path)
    print(f"🗑️ Removed original CSV: {csv_path}")

if __name__ == "__main__":
    download_and_extract_kaggle_data()
