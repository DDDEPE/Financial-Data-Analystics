# hw1_p3_period_definition.py
# Problem 3: Period Definition (+ 터미널 요약 출력, 결과 저장)
# - eom 기준으로 2개 하위기간 라벨링
# - 터미널에 전체/국가별 집계 및 기간별 날짜 범위/미리보기 출력
# - 결과를 CSV/Parquet로 저장

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
# 흔히 사용될 법한 후보 경로들(없으면 --input 지정)
CANDIDATES = [
    DATA_DIR / "hw1_firm_monthly_enhanced.csv",
    DATA_DIR / "hw1_firm_monthly_enhanced.parquet",
    DATA_DIR / "hw1_firm_monthly_base.parquet",
    DATA_DIR / "hw1_firm_monthly_base.csv",
    DATA_DIR / "hw1_country_monthly_returns.csv",
    DATA_DIR / "hw1_country_monthly_returns.parquet",
]

OUT_CSV_DEFAULT = DATA_DIR / "hw1_with_period.csv"
OUT_PQ_DEFAULT  = DATA_DIR / "hw1_with_period.parquet"

def read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    if path.suffix.lower() == ".csv":
        # eom을 날짜형으로 파싱 시도
        return pd.read_csv(path, parse_dates=["eom"], infer_datetime_format=True)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        # eom이 object일 경우 날짜 변환 시도
        if "eom" in df.columns and not np.issubdtype(df["eom"].dtype, np.datetime64):
            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
        return df
    else:
        raise ValueError("지원하지 않는 확장자입니다. csv/parquet만 지원.")

def add_period(df: pd.DataFrame, date_col: str = "eom") -> pd.DataFrame:
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' 컬럼이 없습니다. 입력 데이터에 월말 날짜 컬럼을 포함해 주세요.")
    # 날짜형 보정
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"'{date_col}' 컬럼의 날짜 변환이 모두 실패했습니다. 원 데이터를 확인해 주세요.")

    crisis_start = pd.Timestamp("2020-03-01")
    crisis_end   = pd.Timestamp("2021-12-31")
    recov_start  = pd.Timestamp("2022-01-01")
    recov_end    = pd.Timestamp("2024-12-31")

    eom = df[date_col]
    df["period"] = np.select(
        condlist=[
            (eom >= crisis_start) & (eom <= crisis_end),
            (eom >= recov_start)  & (eom <= recov_end),
        ],
        choicelist=["COVID Crisis", "Post-crisis Recovery"],
        default="Out of Sample",
    )
    return df

def print_overview(df: pd.DataFrame):
    print("\n[CHECK] 전체 행수:", len(df))
    print("[CHECK] 기간 구분 값 빈도:")
    print(df["period"].value_counts(dropna=False).to_string())

    # 기간별 날짜 범위 확인
    print("\n[CHECK] 기간별 날짜 범위 (min~max):")
    rng = df.groupby("period")["eom"].agg(["min", "max"]).sort_index()
    print(rng.to_string())

    # iso3(국가 코드)가 있으면 국가×기간 카운트
    if "iso3" in df.columns:
        print("\n[CHECK] 국가×기간 건수(상위 50개까지 표시):")
        crosstab = (
            df.pivot_table(index="iso3", columns="period", values="eom", aggfunc="count", fill_value=0)
              .sort_index()
        )
        # 너무 길면 상위 50개만
        if len(crosstab) > 50:
            print(crosstab.head(50).to_string())
            print(f"... (총 {len(crosstab)}개 국가)")
        else:
            print(crosstab.to_string())

    # 기간별 미니 미리보기
    print("\n[PREVIEW] 기간별 상위 2행 미리보기:")
    for p in ["COVID Crisis", "Post-crisis Recovery", "Out of Sample"]:
        sub = df.loc[df["period"] == p]
        if len(sub) == 0:
            continue
        cols = ["eom", "period"]
        # 정보성 컬럼을 조금 더 보여주기
        for k in ["iso3", "gvkey", "iid", "ret", "mret", "mktcap", "country_ret_ew", "country_ret_vw"]:
            if k in sub.columns and k not in cols:
                cols.append(k)
        print(f"\n-- {p} ({len(sub)} rows) --")
        print(sub[cols].head(2).to_string(index=False))

def save_outputs(df: pd.DataFrame, out_csv: Path, out_pq: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_pq, index=False)
    print(f"\n[SAVED] CSV : {out_csv}")
    print(f"[SAVED] PQ  : {out_pq}")

def guess_input() -> Path | None:
    for p in CANDIDATES:
        if p.exists():
            return p
    return None

def main():
    parser = argparse.ArgumentParser(description="Problem 3: 하위기간(period) 정의 및 요약 출력")
    parser.add_argument("--input", type=str, default=None, help="입력 파일 경로(csv/parquet). 지정 안하면 data/ 내 후보 자동 탐색")
    parser.add_argument("--out_csv", type=str, default=str(OUT_CSV_DEFAULT), help="출력 CSV 경로")
    parser.add_argument("--out_pq",  type=str, default=str(OUT_PQ_DEFAULT),  help="출력 Parquet 경로")
    args = parser.parse_args()

    in_path = Path(args.input) if args.input else guess_input()
    if in_path is None:
        print("[ERROR] 입력 파일을 찾지 못했습니다. --input 경로를 지정하거나 data/에 파일을 두세요.", file=sys.stderr)
        for c in CANDIDATES:
            print("  - 후보:", c, file=sys.stderr)
        sys.exit(1)

    print("[STEP] 입력 로드:", in_path)
    df = read_any(in_path)

    print("[STEP] 기간(period) 컬럼 생성 (COVID Crisis / Post-crisis Recovery / Out of Sample)")
    df = add_period(df, date_col="eom")

    print_overview(df)

    out_csv = Path(args.out_csv)
    out_pq  = Path(args.out_pq)
    save_outputs(df, out_csv, out_pq)

    print("\n[DONE] Problem 3 완료.")

if __name__ == "__main__":
    main()
