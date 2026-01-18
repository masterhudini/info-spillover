import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bigquery_client import BigQueryClient

def main():
    load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"))

    bq = BigQueryClient(
        project_id=os.getenv("BIGQUERY_PROJECT_ID"),
        dataset_id=os.getenv("BIGQUERY_DATASET_ID")
    )

    query = f"""
    SELECT DISTINCT symbol
    FROM `{bq.project_id}.{bq.dataset_id}.crypto_prices`
    ORDER BY symbol
    """
    df = bq.query_data(query)
    print(", ".join(df["symbol"].tolist()))

if __name__ == "__main__":
    main()
