from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

args = {
    "owner": "admin",
}

dag = DAG(
    dag_id="test_dag",
    default_args=args,
    schedule_interval="0 0 * * *",
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60),
    # tags=[],
    # params={},
)

test = BashOperator(
    task_id="test_task",
    bash_command="echo test_dag",
    dag=dag,
)

if __name__ == "__main__":
    dag.cli()
