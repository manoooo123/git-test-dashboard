"""
Pearls AQI Predictor — Apache Airflow DAG (v2.1.0)

Pipeline structure:
  Hourly DAG  : Feature ingestion only (OpenAQ + Open-Meteo → Feature Store)
  Daily DAG   : Model retraining + SHAP re-generation

Two separate DAGs are defined to avoid the previous bug where model retraining
ran every hour (expensive and unnecessary). Feature refresh runs hourly;
model retraining runs once per day.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# ----------------------------------------------------------------------------
# Shared default args
# ----------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "owner": "pearls_mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ============================================================================
# DAG 1: Hourly Feature Ingestion
# Runs every hour. Fetches live data, engineers features, writes to Feature Store.
# Does NOT retrain models.
# ============================================================================

with DAG(
    dag_id="pearls_aqi_hourly_feature_pipeline",
    default_args=_DEFAULT_ARGS,
    description="Hourly AQI feature ingestion: OpenAQ v3 + Open-Meteo → Feature Store",
    schedule_interval="0 * * * *",      # Every hour at :00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,                  # Prevent overlapping runs
    tags=["mlops", "aqi", "pearls", "feature-pipeline"],
) as hourly_dag:

    hourly_feature_ingestion = BashOperator(
        task_id="hourly_feature_ingestion",
        bash_command=(
            "cd ${AIRFLOW_HOME}/pearls_aqi && "
            "python feature_pipeline/daily_live_refresh.py"
        ),
        do_xcom_push=False,
    )


# ============================================================================
# DAG 2: Daily Model Retraining + SHAP
# Runs once per day. Retrieves features from store, retrains all models,
# evaluates, selects best, re-generates feature importances.
# ============================================================================

with DAG(
    dag_id="pearls_aqi_daily_training_pipeline",
    default_args=_DEFAULT_ARGS,
    description="Daily ML model retraining and SHAP re-generation for Pearls AQI",
    schedule_interval="0 1 * * *",     # Daily at 01:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "aqi", "pearls", "training"],
) as daily_dag:

    daily_model_retraining = BashOperator(
        task_id="daily_model_retraining",
        bash_command=(
            "cd ${AIRFLOW_HOME}/pearls_aqi && "
            "python training_pipeline/train_3cities.py"
        ),
        do_xcom_push=False,
    )

    daily_shap_analysis = BashOperator(
        task_id="daily_shap_feature_importance",
        bash_command=(
            "cd ${AIRFLOW_HOME}/pearls_aqi && "
            "python explainability/shap_analysis.py"
        ),
        do_xcom_push=False,
    )

    # Retraining must complete before SHAP (SHAP uses the newly trained models)
    daily_model_retraining >> daily_shap_analysis
