from dagster import repository

from dags.assets.data_ingestion import downloading_kaggle_dataset, scrapping_youtube_data, aggregated_data
from dags.assets.data_annotation import auto_annotation
from dags.assets.data_processing import anomaly_cleaning, cleaning

from dags.jobs.active_learning_job import active_learning_job
from dags.jobs.data_jobs import data_ingestion_job, data_manipulation_job

@repository
def my_repo():
    return [
        downloading_kaggle_dataset,
        scrapping_youtube_data,
        aggregated_data,
        anomaly_cleaning,
        auto_annotation,
        cleaning,
        active_learning_job,
        data_ingestion_job,
        data_manipulation_job
    ]


