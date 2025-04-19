from dags.ops.active_learning import active_learning_round
from dagster import job

@job
def active_learning_job():
    active_learning_round()