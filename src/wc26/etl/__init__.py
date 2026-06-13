# ETL: validate raw data, drive feature engineering, and load into the warehouse.
#
# Thin orchestration over ``wc26.features``; exposes plain-Python task functions
# that both the Airflow DAG and the GitHub Actions pipeline call.
