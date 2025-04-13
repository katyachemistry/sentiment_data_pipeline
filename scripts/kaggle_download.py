import gdown
import zipfile
import os

def download_and_extract_kaggle_data(
    url: str = 'https://drive.google.com/uc?id=1QDHZrqQ8gbNW18Flxs5RCsC_tv7hxfeH',
    output_path: str = 'data/raw/sentiment_kaggle.zip',
    extract_dir: str = 'data/raw/sentiment_kaggle'
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"⬇️ Downloading Kaggle sentiment140 dataset from Google Drive...")
    gdown.download(url, output_path, quiet=False)

    print(f"📦 Extracting to: {extract_dir}")
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(output_path)

    print("✅ Download and extraction complete.")

if __name__ == "__main__":
    download_and_extract_kaggle_data()
