import pandas as pd

def safe_save_outputs(df: pd.DataFrame, csv_path, pq_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("Saved CSV:", csv_path.resolve())
    try:
        import pyarrow  # noqa
        df.to_parquet(pq_path, index=False, engine="pyarrow")
        print("Saved Parquet:", pq_path.resolve())
    except Exception:
        try:
            import fastparquet  # noqa
            df.to_parquet(pq_path, index=False, engine="fastparquet")
            print("Saved Parquet:", pq_path.resolve())
        except Exception:
            print("Parquet 엔진(pyarrow/fastparquet) 미설치 → Parquet 저장 생략")
