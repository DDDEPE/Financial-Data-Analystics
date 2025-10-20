import sys
from hw1_pipeline.config import DATA_DIR, TARGET_ISO3
from hw1_pipeline.wrds_utils import connect_wrds
from hw1_pipeline.pipeline import load_firm_monthly
from hw1_pipeline.io_utils import safe_save_outputs
from hw1_pipeline.progress import iter_progress

def main():
    assert TARGET_ISO3 == {"AUS","FRA","DEU","JPN","GBR","BRA","CHN","IND","ZAF","TUR"}, \
        f"TARGET_ISO3이 지정한 10개와 다릅니다: {TARGET_ISO3}"

    db = connect_wrds()

    # 상위 단계는 pipeline 내부 step() 로그로 충분하지만, 확장 대비 예시 유지
    steps = [("기업 월별 데이터 로드 & mktcap 구성 (10개국 엄격)", load_firm_monthly)]
    for label, fn in iter_progress(steps, desc="[pipeline] steps", total=len(steps)):
        print(f"\n== {label} ==")
        firm = fn(db)

    out_csv = DATA_DIR / "hw1_firm_monthly_base.csv"
    out_pq  = DATA_DIR / "hw1_firm_monthly_base.parquet"
    safe_save_outputs(firm, out_csv, out_pq)

    print("\n=== 요약(국가별 관측치/기업수) ===")
    g = (firm.groupby("country")
              .agg(n_obs=("mret","count"),
                   n_firms=("gvkey", "nunique"))
              .reset_index()
              .sort_values("country"))
    try:
        import pandas as pd
        pd.set_option("display.max_rows", None)
    except Exception:
        pass
    print(g.to_string(index=False))

if __name__ == "__main__":
    sys.exit(main())
