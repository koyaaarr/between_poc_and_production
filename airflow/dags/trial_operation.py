from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

args = {
    "owner": "admin",
}

dag = DAG(
    dag_id="trial_operation",
    default_args=args,
    schedule_interval="0 0 * * *",
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60),
    # tags=[],
    # params={},
)

accumulate = BashOperator(
    task_id="accumulate_train_data",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/accmulate.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/accmulate.ipynb -p TARGET_DATE 20210101",
    dag=dag,
)

feat_eng_train = BashOperator(
    task_id="feature_engineering_train_data",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/feature_engineering.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/feature_engineering.ipynb -p TARGET_DATE 20210101 -p DATA_TYPE train",
    dag=dag,
)

feat_eng_test = BashOperator(
    task_id="feature_engineering_test_data",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/feature_engineering.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/feature_engineering.ipynb -p TARGET_DATE 20210101 -p DATA_TYPE test",
    dag=dag,
)

learn = BashOperator(
    task_id="learn",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/learn.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/learn.ipynb -p TARGET_DATE 20210101",
    dag=dag,
)

inference = BashOperator(
    task_id="inference",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/inference.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/inference.ipynb -p TARGET_DATE 20210101",
    dag=dag,
)

accumulate >> feat_eng_train >> learn

[learn, feat_eng_test] >> inference

if __name__ == "__main__":
    dag.cli()
