from dagster import asset, AssetExecutionContext
from pathlib import Path
import pandas as pd

from scripts.kaggle_download import download_and_extract_kaggle_data
from scripts.youtube_scrapping import parse_youtube_comments
from scripts.aggregate_dataframe import aggregate_datasets

@asset(group_name="ingestion")
def downloading_kaggle_dataset(context: AssetExecutionContext) -> pd.DataFrame:
    """Kaggle dataset download"""
    try:
        df = download_and_extract_kaggle_data()
        context.log.info("✅ Kaggle dataset downloaded and extracted successfully.")
        return df
    except Exception as e:
        context.log.info(f"❌ Error downloading Kaggle dataset: {str(e)}")
        raise


@asset(group_name="ingestion")
def scrapping_youtube_data(context: AssetExecutionContext) -> pd.DataFrame:
    """YouTube data scraping"""
    current_dir = Path(__file__).resolve().parent
    output_path = current_dir.parent / "data" / "raw" / "youtube_comments_df.parquet"
    context.log.info(f"Checking for file at: {output_path}")
    if output_path.exists():
        context.log.info(f"📁 {output_path} already exists. Skipping scraping.")
        return pd.read_parquet(output_path)
    try:
        df = parse_youtube_comments()
        context.log.info("✅ YouTube data scraped and saved successfully.")
        return df
    except Exception as e:
        context.log.info(f"❌ Error scraping YouTube data: {str(e)}")
        raise


@asset(group_name="ingestion", deps=[downloading_kaggle_dataset, scrapping_youtube_data])
def aggregated_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Aggregating dataset"""
    try:
        df = aggregate_datasets()  # This returns a DataFrame
        output_path = Path("data/aggregated_dataset.parquet")

        context.add_output_metadata({
            "path": str(output_path),
            "size (MB)": f"{output_path.stat().st_size / (1024*1024):.2f}"        })

        context.log.info(f"✅ Aggregated dataset saved to {output_path}")
        return df
    except Exception as e:
        context.log.info(f"❌ Error aggregating datasets: {str(e)}")
        raise