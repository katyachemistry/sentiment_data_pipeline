from dagster import asset, AssetExecutionContext, get_dagster_logger
from pathlib import Path
from scripts.preprocess import process_and_save_data
import pandas as pd

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

@asset(group_name="preprocessing", deps=[aggregated_data])
def preprocess_data(context: AssetExecutionContext) -> str:
    """Asset representing the data preprocessing step"""
    logger = get_dagster_logger()
    current_dir = Path(__file__).resolve().parent
    output_file_path = current_dir.parent / "data" / "cleaned_dataset.csv" 
    if output_file_path.exists():
        logger.info("📁 'cleaned_dataset.csv' already exists. Skipping scraping.")
        return str(output_file_path)

    try:
        # Define file paths for the input and output CSVs
        input_file_path = current_dir.parent / "data" / "aggregated_dataset.csv" 

        # Log the preprocessing process
        logger.info(f"Starting preprocessing on {input_file_path} and saving to {output_file_path}")

        # Call the function from the preprocessing script to process the data
        process_and_save_data(input_file_path, output_file_path)

        # Log success and return the path of the cleaned dataset
        context.add_output_metadata({
            "path": output_file_path,
            "size (MB)": f"{Path(output_file_path).stat().st_size / (1024*1024):.2f}",
            "num_rows": len(pd.read_csv(output_file_path)),
        })

        logger.info(f"✅ Data preprocessing completed and saved to {output_file_path}")
        return str(output_file_path)
    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {str(e)}")
        raise

