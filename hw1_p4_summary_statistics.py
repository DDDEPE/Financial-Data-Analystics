# hw1_p4_summary_statistics.py
# Problem 4: Summary Statistics by Country and Period (총수정본)
# - 입력: 국가 월별 수익률 (eom, iso3, [EW, VW_BOM, VW_EOM], [period])
# - period 미존재 시 문제3 규칙으로 생성
# - 컬럼 자동 매핑:
#     EW       -> country_ret_ew
#     VW_BOM   -> country_ret_vw (기본)
#     VW_EOM   -> country_ret_vw (옵션 --vw_source VW_EOM)
# - 산출: 국가×기간별 요약통계(Mean, Median, Std, Min, Max, AC(1), Skew, Excess Kurtosis)
# - 터미널 전부 출력 + CSV/Parquet 저장

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
CANDIDATES = [
    DATA_DIR / "hw1_country_monthly_returns.csv",
    DATA_DIR / "hw1_country_monthly_returns.parquet",
    DATA_DIR / "hw1_with_period.csv",
    DATA_DIR / "hw1_with_period.parquet",
]

OUT_DIR = DATA_DIR
OUT_EW_CSV = OUT_DIR / "p4_stats_ew.csv"
OUT_VW_CSV = OUT_DIR / "p4_stats_vw.csv"
OUT_ALL_CSV = OUT_DIR / "p4_stats_all.csv"
OUT_EW_PQ  = OUT_DIR / "p4_stats_ew.parquet"
OUT_VW_PQ  = OUT_DIR / "p4_stats_vw.parquet"
OUT_ALL_PQ = OUT_DIR / "p4_stats_all.parquet"

CRISIS_START = pd.Timestamp("2020-03-01")
CRISIS_END   = pd.Timestamp("2021-12-31")
RECOV_START  = pd.Timestamp("2022-01-01")
RECOV_END    = pd.Timestamp("2024-12-31")

def read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    if path.suffix.lower() == ".csv":
        # infer_datetime_format 인자는 더 이상 필요 없음
        df = pd.read_csv(path, parse_dates=["eom"])
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        if "eom" in df.columns and not np.issubdtype(df["eom"].dtype, np.datetime64):
            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
    else:
        raise ValueError("csv/parquet만 지원합니다.")
    return df

def ensure_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "period" in df.columns:
        return df
    eom = pd.to_datetime(df["eom"], errors="coerce")
    df["period"] = np.select(
        [
            (eom >= CRISIS_START) & (eom <= CRISIS_END),
            (eom >= RECOV_START)  & (eom <= RECOV_END),
        ],
        ["COVID Crisis", "Post-crisis Recovery"],
        default="Out of Sample",
    )
    return df

def alias_columns(df: pd.DataFrame, vw_source: str = "VW_BOM") -> pd.DataFrame:
    """
    데이터의 실제 컬럼명을 p4에서 쓰는 표준 이름(country_ret_ew, country_ret_vw)으로 맞춥니다.
    - EW -> country_ret_ew
    - (vw_source) -> country_ret_vw  (VW_BOM 또는 VW_EOM)
    """
    df = df.copy()
    # EW 매핑
    if "country_ret_ew" not in df.columns:
        if "EW" in df.columns:
            df["country_ret_ew"] = df["EW"]
        else:
            # 없는 경우는 생략(나중에 has_ew False 처리)
            pass

    # VW 매핑
    vw_source = vw_source.upper()
    cand = []
    if vw_source == "VW_EOM":
        cand = ["VW_EOM", "VW_BOM"]  # EOM 우선, 없으면 BOM fallback
    else:
        cand = ["VW_BOM", "VW_EOM"]  # BOM 우선, 없으면 EOM fallback

    if "country_ret_vw" not in df.columns:
        for c in cand:
            if c in df.columns:
                df["country_ret_vw"] = df[c]
                break
        # 그래도 없으면 아무것도 안 만듦(나중에 has_vw False 처리)

    return df

def _excess_kurtosis(x: pd.Series) -> float:
    # pandas.Series.kurt()는 excess(=Fisher)입니다.
    return float(pd.Series(x).kurt())

def _ac1(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return np.nan
    try:
        return float(x.autocorr(lag=1))
    except Exception:
        return np.nan

def compute_stats(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    iso3 × period 그룹별로 value_col의 요약통계를 계산합니다.
    """
    use = df[["iso3", "period", "eom", value_col]].copy()
    use[value_col] = pd.to_numeric(use[value_col], errors="coerce")
    use = use.dropna(subset=[value_col])

    def _summ(g: pd.DataFrame) -> pd.Series:
        x = g[value_col]
        n = int(x.notna().sum())
        return pd.Series({
            "n": n,
            "mean": float(x.mean()) if n else np.nan,
            "median": float(x.median()) if n else np.nan,
            "std": float(x.std(ddof=1)) if n >= 2 else np.nan,
            "min": float(x.min()) if n else np.nan,
            "max": float(x.max()) if n else np.nan,
            "ac1": _ac1(x),
            "skew": float(x.skew()) if n >= 3 else np.nan,
            "excess_kurtosis": _excess_kurtosis(x) if n >= 4 else np.nan,
        })

    out = (use
           .groupby(["iso3", "period"], dropna=False, sort=True)
           .apply(_summ)
           .reset_index())
    out.insert(2, "series", value_col)  # 구분(ew/vw)
    return out

def pretty_print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_overview(df: pd.DataFrame):
    pretty_print_header("[개요] 입력 데이터 확인")
    print("전체 행수:", len(df))
    print("컬럼 목록:", list(df.columns))
    for col in ["eom", "iso3", "period", "country_ret_ew", "country_ret_vw", "EW", "VW_BOM", "VW_EOM"]:
        print(f" - 보유 여부 [{col}]:", col in df.columns)
    if "period" in df.columns:
        print("\n[분포] period 빈도:")
        print(df["period"].value_counts(dropna=False).to_string())

    if "iso3" in df.columns and "period" in df.columns:
        print("\n[표본수] 국가×기간 카운트(상위 50개):")
        cnt = (df
               .groupby(["iso3", "period"])["eom"]
               .count()
               .rename("n")
               .reset_index()
               .sort_values(["iso3", "period"]))
        if len(cnt) > 50:
            print(cnt.head(50).to_string(index=False))
            print(f"... (총 {len(cnt)} 개 조합)")
        else:
            print(cnt.to_string(index=False))

def save_all(ew: pd.DataFrame | None, vw: pd.DataFrame | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if ew is not None:
        ew.to_csv(OUT_EW_CSV, index=False)
        ew.to_parquet(OUT_EW_PQ, index=False)
        print(f"\n[SAVED] EW  CSV: {OUT_EW_CSV}")
        print(f"[SAVED] EW  PQ : {OUT_EW_PQ}")
    if vw is not None:
        vw.to_csv(OUT_VW_CSV, index=False)
        vw.to_parquet(OUT_VW_PQ, index=False)
        print(f"[SAVED] VW  CSV: {OUT_VW_CSV}")
        print(f"[SAVED] VW  PQ : {OUT_VW_PQ}")
    if ew is not None and vw is not None:
        all_df = pd.concat([ew, vw], axis=0, ignore_index=True)
        all_df.to_csv(OUT_ALL_CSV, index=False)
        all_df.to_parquet(OUT_ALL_PQ, index=False)
        print(f"[SAVED] ALL CSV: {OUT_ALL_CSV}")
        print(f"[SAVED] ALL PQ : {OUT_ALL_PQ}")

def guess_input() -> Path | None:
    for p in CANDIDATES:
        if p.exists():
            return p
    return None

def main():
    parser = argparse.ArgumentParser(description="Problem 4: 국가·기간별 요약통계 계산 및 터미널 출력")
    parser.add_argument("--input", type=str, default=None, help="입력 파일(csv/parquet). 미지정 시 data/ 후보 자동탐색")
    parser.add_argument("--vw_source", type=str, default="VW_BOM", choices=["VW_BOM", "VW_EOM"],
                        help="VW 시리즈 소스 선택 (기본: VW_BOM)")
    args = parser.parse_args()

    in_path = Path(args.input) if args.input else guess_input()
    if in_path is None:
        print("[ERROR] 입력 파일을 찾지 못했습니다. --input으로 지정하거나 data/ 폴더에 두세요.", file=sys.stderr)
        for c in CANDIDATES:
            print("  - 후보:", c, file=sys.stderr)
        sys.exit(1)

    print("[STEP] 입력 로드:", in_path)
    df = read_any(in_path)

    # 필수 컬럼 체크 및 보정
    need_cols = {"eom", "iso3"}
    if not need_cols.issubset(df.columns):
        missing = need_cols - set(df.columns)
        raise KeyError(f"필수 컬럼 누락: {missing}. (eom, iso3 필요)")
    df = ensure_period(df)

    # ★ 컬럼 자동 매핑: EW/VW_* → country_ret_ew / country_ret_vw
    df = alias_columns(df, vw_source=args.vw_source)

    # 개요 출력
    print_overview(df)

    # 시리즈 존재 여부
    has_ew = "country_ret_ew" in df.columns
    has_vw = "country_ret_vw" in df.columns

    if not (has_ew or has_vw):
        print("\n[ERROR] 사용할 수 있는 수익률 컬럼이 없습니다.", file=sys.stderr)
        print(" - 가능 컬럼: EW, VW_BOM, VW_EOM 중 최소 하나 필요", file=sys.stderr)
        sys.exit(2)

    def compute_and_print(tag: str, series_col: str):
        pretty_print_header(f"[통계] {tag} 국가 월별 수익률")
        stats = compute_stats(df, series_col)
        stats = stats.sort_values(["iso3", "period"]).reset_index(drop=True)
        print(stats.to_string(index=False, max_rows=2000))
        return stats

    ew_stats = compute_and_print("Equally Weighted (EW)", "country_ret_ew") if has_ew else None
    vw_stats = compute_and_print(f"Value Weighted ({args.vw_source})", "country_ret_vw") if has_vw else None

    # 저장
    save_all(ew_stats, vw_stats)

    # 비교용 피벗 출력
    if ew_stats is not None:
        pretty_print_header("[피벗] EW — 국가×기간 행, 통계치 열")
        ew_wide = (ew_stats
                   .set_index(["iso3", "period"])
                   [["n", "mean", "median", "std", "min", "max", "ac1", "skew", "excess_kurtosis"]]
                   .sort_index())
        print(ew_wide.head(60).to_string())
        if len(ew_wide) > 60:
            print(f"... ({len(ew_wide)} 행)")

    if vw_stats is not None:
        pretty_print_header(f"[피벗] {args.vw_source} — 국가×기간 행, 통계치 열")
        vw_wide = (vw_stats
                   .set_index(["iso3", "period"])
                   [["n", "mean", "median", "std", "min", "max", "ac1", "skew", "excess_kurtosis"]]
                   .sort_index())
        print(vw_wide.head(60).to_string())
        if len(vw_wide) > 60:
            print(f"... ({len(vw_wide)} 행)")

    print("\n[DONE] Problem 4 완료.")

if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    main()
