from dagster import asset, AssetExecutionContext, get_dagster_logger, repository
from pathlib import Path

# Direct function imports
from scripts.kaggle_download import download_and_extract_kaggle_data
from scripts.youtube_scrapping import parse_youtube_comments
from scripts.aggregate_dataframe import aggregate_datasets

@asset(group_name="ingestion")
def kaggle_dataset(context: AssetExecutionContext) -> bool:
    """Asset representing the Kaggle dataset download"""
    logger = get_dagster_logger()
    try:
        download_and_extract_kaggle_data()
        logger.info("✅ Kaggle dataset downloaded and extracted successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error downloading Kaggle dataset: {str(e)}")
        raise


@asset(group_name="ingestion")
def youtube_data(context: AssetExecutionContext) -> bool:
    """Asset representing the YouTube data scraping"""
    logger = get_dagster_logger()
    current_dir = Path(__file__).resolve().parent
    output_path = current_dir.parent / "data" / "raw" / "youtube_comments_df.csv"

    logger.info(f"Checking for file at: {output_path}")
    if output_path.exists():
        logger.info("📁 'youtube_comments_df.csv' already exists. Skipping scraping.")
        return True

    try:
        parse_youtube_comments()
        logger.info("✅ YouTube data scraped and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error scraping YouTube data: {str(e)}")
        raise


@asset(group_name="ingestion", deps=[kaggle_dataset, youtube_data])
def aggregated_data(context: AssetExecutionContext) -> str:
    """Asset representing the aggregated dataset"""
    logger = get_dagster_logger()
    try:
        df = aggregate_datasets()  # This returns a DataFrame
        output_path = Path("data/aggregated_dataset.csv")

        context.add_output_metadata({
            "path": str(output_path),
            "size (MB)": f"{output_path.stat().st_size / (1024*1024):.2f}",
            "num_rows": len(df),
        })

        logger.info(f"✅ Aggregated dataset saved to {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"❌ Error aggregating datasets: {str(e)}")
        raise
