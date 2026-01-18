import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bigquery_client import BigQueryClient


def main():
    load_dotenv(dotenv_path="/home/Hudini/projects/info_spillover_enhanced/.env")

    bq = BigQueryClient(
        project_id=os.getenv("BIGQUERY_PROJECT_ID"),
        dataset_id=os.getenv("BIGQUERY_DATASET_ID"),
        sentiment_table=os.getenv("BIGQUERY_SENTIMENT_TABLE")
    )

    start_date = os.getenv("START_DATE")
    end_date = os.getenv("END_DATE")

    df = bq.get_post_sentiment_aggregation(start_date, end_date)
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'post_created_utc'})

    df['post_created_utc'] = pd.to_datetime(df['post_created_utc'])

    sub_counts = df['subreddit'].value_counts()
    print("Top subreddits by row count (5-min bins):")
    print(sub_counts.head(15))

    freqs = ["5T", "15T", "30T", "1H", "2H", "4H", "6H", "12H", "1D"]

    for freq in freqs:
        print(f"\n=== Overlap analysis @ {freq} ===")
        df_res = df.copy()
        df_res['ts'] = df_res['post_created_utc'].dt.floor(freq)

        per_sub = df_res.groupby(['subreddit', 'ts']).size().reset_index(name='n')

        for n in [5, 10, 15, 20]:
            subs = list(sub_counts.head(n).index)
            pivot = per_sub[per_sub['subreddit'].isin(subs)].pivot_table(
                index='ts', columns='subreddit', values='n', aggfunc='size'
            )
            common = pivot.dropna().shape[0]
            total = pivot.shape[0]
            print(f"Top {n}: common_timestamps={common} / total={total}")


if __name__ == "__main__":
    main()
