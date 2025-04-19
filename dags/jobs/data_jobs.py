import dagster as dg

data_ingestion_job = dg.define_asset_job(
    name="data_ingestion_job", selection=["downloading_kaggle_dataset", "scrapping_youtube_data", "aggregated_data"]
)

data_manipulation_job = dg.define_asset_job(
    name="data_manipulation_job", selection=["anomaly_cleaning", "auto_annotation", "cleaning"]
)


