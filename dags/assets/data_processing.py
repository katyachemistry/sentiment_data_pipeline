from dagster import asset, AssetExecutionContext
from pathlib import Path
import pandas as pd

from scripts.annotation import *
from scripts.preprocess import *
from dags.assets.data_ingestion import aggregated_data
from dags.assets.data_annotation import auto_annotation

@asset(group_name="preprocessing", deps=[aggregated_data])
def anomaly_cleaning(context: AssetExecutionContext) -> pd.DataFrame:
    """Auto-annotation, complexity analysis and selection of data for manual annotation (uncertainty threshold 0.6)"""
    try:
        clean_data = detect_anomalies(Path("data/aggregated_dataset.parquet"))
        context.log.info(f"Anomaly cleaning successfull")
        return clean_data
    except Exception as e:
        context.log.info(f"❌ Error during anomaly cleaning: {str(e)}")
        raise

@asset(group_name="preprocessing", deps=[anomaly_cleaning, auto_annotation])
def cleaning(context: AssetExecutionContext, anomaly_cleaning: pd.DataFrame, auto_annotation: pd.DataFrame) -> pd.DataFrame:
    """Combining labelled data. Cleaning sfrom emojis, numbers, duplicates..."""
    try:
        data = load_data_for_preprocessing(
            anomaly_cleaning,
            auto_annotation,
            'data/manual_annotation/project-2-at-2025-04-17-17-52-4348a69f.csv'
        )
        result = process_and_save_data(data)
        context.log.info(f"Sucessfully cleaned the data.")
        return result
    except Exception as e:
        context.log.info(f"❌ Error during preprocessing: {str(e)}")
        raise