from dagster import asset, AssetExecutionContext, op, job, repository, In, Out, String
from pathlib import Path
import pandas as pd

from scripts.kaggle_download import download_and_extract_kaggle_data
from scripts.youtube_scrapping import parse_youtube_comments
from scripts.aggregate_dataframe import aggregate_datasets
from scripts.annotation import *
from scripts.preprocess import *
from scripts.active_learning import load_data, run_active_learning

@asset(group_name="ingestion")
def kaggle_dataset(context: AssetExecutionContext) -> bool:
    """Kaggle dataset download"""
    try:
        download_and_extract_kaggle_data()
        context.log.info("✅ Kaggle dataset downloaded and extracted successfully.")
        return True
    except Exception as e:
        context.log.info(f"❌ Error downloading Kaggle dataset: {str(e)}")
        raise


@asset(group_name="ingestion")
def youtube_data(context: AssetExecutionContext) -> bool:
    """YouTube data scraping"""
    current_dir = Path(__file__).resolve().parent
    output_path = current_dir.parent / "data" / "raw" / "youtube_comments_df.parquet"
    context.log.info(f"Checking for file at: {output_path}")
    if output_path.exists():
        context.log.info(f"📁 {output_path} already exists. Skipping scraping.")
        return True
    try:
        parse_youtube_comments()
        context.log.info("✅ YouTube data scraped and saved successfully.")
        return True
    except Exception as e:
        context.log.info(f"❌ Error scraping YouTube data: {str(e)}")
        raise


@asset(group_name="ingestion", deps=[kaggle_dataset, youtube_data])
def aggregated_data(context: AssetExecutionContext) -> bool:
    """Aggregating dataset"""
    try:
        aggregate_datasets()  # This returns a DataFrame
        output_path = Path("data/aggregated_dataset.parquet")

        context.add_output_metadata({
            "path": str(output_path),
            "size (MB)": f"{output_path.stat().st_size / (1024*1024):.2f}"        })

        context.log.info(f"✅ Aggregated dataset saved to {output_path}")
        return True
    except Exception as e:
        context.log.info(f"❌ Error aggregating datasets: {str(e)}")
        raise

@asset(group_name="autom_annotation", deps=[aggregated_data])
def anomaly_cleaning(context: AssetExecutionContext) -> str:
    """Auto-annotation, complexity analysis and selection of data for manual annotation (uncertainty threshold 0.6)"""
    try:
        clean_data = detect_anomalies(Path("data/aggregated_dataset.parquet"))
        context.log.info(f"Anomaly cleaning successfull")
        return clean_data
    except Exception as e:
        context.log.info(f"❌ Error during anomaly cleaning: {str(e)}")
        raise

@asset(group_name="autom_annotation")
def auto_annotation(context: AssetExecutionContext, anomaly_cleaning: str) -> bool:
    """Auto-annotation, complexity analysis and selection of data for manual annotation (uncertainty threshold 0.6)"""
    try:
        clean_data = anomaly_cleaning
        annotated_data = auto_annotate(clean_data)
        analyze_complexity(annotated_data)
        context.log.info(f"Auto-annotation, complexity analysis and selection of data for manual annotation complete")
        return True
    except Exception as e:
        context.log.info(f"❌ Error during annotation: {str(e)}")
        raise

@asset(group_name="preprocessing")
def cleaning(context: AssetExecutionContext) -> pd.DataFrame:
    """Combining labelled data. Cleaning from emojis, numbers, duplicates..."""
    try:
        data = load_data(
            'data/no_anomaly_data.parquet',
            'data/auto_annotated_confident.csv',
            'data/manual_annotation/project-2-at-2025-04-17-17-52-4348a69f.csv'
        )
        result = process_and_save_data(data)
        context.log.info(f"Sucessfully cleaned the data.")
        return result
    except Exception as e:
        context.log.info(f"❌ Error during preprocessing: {str(e)}")
        raise


@op(ins={"batch_num": In(int), "new_labels_path": In(String), "n_queries": In(int), "save_model": In(bool)}, out=Out(bool))
def active_learning_round(context, batch_num: int, new_labels_path: str, n_queries: int, save_model:bool) -> bool:
    try:
        if batch_num == 0:
            context.log.info("🔁 First launch of active learning.")
            data = load_data(first_launch=True)
        else:
            context.log.info(f"🔁 Loading data with new labels from: {new_labels_path}")
            data = load_data(first_launch=False, new_labels_path=new_labels_path)

        run_active_learning(data, batch_num=batch_num, n_queries=n_queries, save_model=save_model)
        context.log.info(f"✅ Active learning round {batch_num} complete.")
        return True
    except Exception as e:
        context.log.error(f"❌ Failed active learning round {batch_num}: {str(e)}")
        raise

@job
def active_learning_job():
    active_learning_round()

@repository
def my_repo():
    return [
        kaggle_dataset,
        youtube_data,
        aggregated_data,
        anomaly_cleaning,
        auto_annotation,
        cleaning,
        active_learning_job,
    ]


