from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

args = {
    "owner": "admin",
}


dag = DAG(
    dag_id="trial_operation",
    default_args=args,
    schedule_interval=None,  # スケジュール実行したい場合は"0 0 0 * *"のようにcron式で定義する
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60),
    # tags=[],
    params={
        "target_date": "20210101",
        "work_dir": "~/Code/between_poc_and_production/notebooks/trial/",
    },
)

accumulate = BashOperator(
    task_id="accumulate_train_data",
    bash_command="papermill --cwd {{ dag_run.conf['work_dir'] }} {{ dag_run.conf['work_dir'] }}accmulate.ipynb {{ dag_run.conf['work_dir'] }}logs/accmulate.ipynb -p TARGET_DATE {{ dag_run.conf['target_date'] }}",
    dag=dag,
)

feat_eng_train = BashOperator(
    task_id="feature_engineering_train_data",
    bash_command="papermill --cwd {{ dag_run.conf['work_dir'] }} {{ dag_run.conf['work_dir'] }}feature_engineering.ipynb {{ dag_run.conf['work_dir'] }}logs/feature_engineering.ipynb -p TARGET_DATE {{ dag_run.conf['target_date'] }} -p DATA_TYPE train",
    dag=dag,
)

feat_eng_test = BashOperator(
    task_id="feature_engineering_test_data",
    bash_command="papermill --cwd {{ dag_run.conf['work_dir'] }} {{ dag_run.conf['work_dir'] }}feature_engineering.ipynb {{ dag_run.conf['work_dir'] }}logs/feature_engineering.ipynb -p TARGET_DATE {{ dag_run.conf['target_date'] }} -p DATA_TYPE test",
    dag=dag,
)

learn = BashOperator(
    task_id="learn",
    bash_command="papermill --cwd {{ dag_run.conf['work_dir'] }} {{ dag_run.conf['work_dir'] }}learn.ipynb {{ dag_run.conf['work_dir'] }}logs/learn.ipynb -p TARGET_DATE {{ dag_run.conf['target_date'] }}",
    dag=dag,
)

inference = BashOperator(
    task_id="inference",
    bash_command="papermill --cwd {{ dag_run.conf['work_dir'] }} {{ dag_run.conf['work_dir'] }}inference.ipynb {{ dag_run.conf['work_dir'] }}logs/inference.ipynb -p TARGET_DATE {{ dag_run.conf['target_date'] }}",
    dag=dag,
)

accumulate >> feat_eng_train >> learn

[learn, feat_eng_test] >> inference

if __name__ == "__main__":
    dag.cli()
