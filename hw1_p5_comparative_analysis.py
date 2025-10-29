# hw1_p5_comparative_analysis.py
# Problem 5: Comparative Analysis — Developed vs Emerging (총수정본)
# - 입력 우선순위:
#   (1) data/p4_stats_all.(parquet|csv)  ← 문제4 결과(권장)
#   (2) data/hw1_country_monthly_returns.(parquet|csv) ← 없으면 원시 국가수익률에서 재계산
# - 비교 축:
#   * 그룹: Developed vs Emerging (MSCI, 과제 지시 10개국 고정)
#   * 기간: COVID Crisis vs Post-crisis Recovery (기본)
#   * 옵션: --overall 로 기간 무시(전체 표본) 비교 가능
# - VW 시리즈 선택 옵션: --vw_source (기본 VW_BOM, 대안 VW_EOM; p4 결과 없을 때만 영향)
# - 출력(모두 터미널 인쇄 + CSV/Parquet 저장):
#   (a) 그룹 평균 통계 (group × period × series)
#   (b) Emerging - Developed 차이표 + 핵심 비교 라인
#   (c) 회복기 변화 (Recovery - Crisis) 및 변화의 그룹 간 차이 Δ(EME) - Δ(DEV)

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np

DATA_DIR = Path("data")

# ===== 입력 후보 =====
P4_ALL = [DATA_DIR / "p4_stats_all.parquet", DATA_DIR / "p4_stats_all.csv"]
P2_RAW = [DATA_DIR / "hw1_country_monthly_returns.parquet", DATA_DIR / "hw1_country_monthly_returns.csv"]

# ===== 분류: 과제 지시 10개국 =====
DEVELOPED = {"GBR", "DEU", "JPN", "FRA", "AUS"}
EMERGING  = {"CHN", "IND", "BRA", "ZAF", "TUR"}

CRISIS_START = pd.Timestamp("2020-03-01")
CRISIS_END   = pd.Timestamp("2021-12-31")
RECOV_START  = pd.Timestamp("2022-01-01")
RECOV_END    = pd.Timestamp("2024-12-31")

OUT_SUMMARY_CSV = DATA_DIR / "p5_group_summary.csv"
OUT_SUMMARY_PQ  = DATA_DIR / "p5_group_summary.parquet"
OUT_DIFFS_CSV   = DATA_DIR / "p5_group_diffs.csv"
OUT_DIFFS_PQ    = DATA_DIR / "p5_group_diffs.parquet"
OUT_CHG_CSV     = DATA_DIR / "p5_group_changes.csv"
OUT_CHG_PQ      = DATA_DIR / "p5_group_changes.parquet"

# ========= 유틸 =========

def _read_first(paths):
    for p in paths:
        if p.exists():
            if p.suffix.lower() == ".parquet":
                return pd.read_parquet(p), p
            return pd.read_csv(p), p
    return None, None

def _ensure_period(df: pd.DataFrame) -> pd.DataFrame:
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

def _alias_from_raw(df: pd.DataFrame, vw_source: str = "VW_BOM") -> pd.DataFrame:
    """
    raw(country monthly) → 표준 이름 매핑:
      EW -> country_ret_ew
      (vw_source) -> country_ret_vw  (VW_BOM 기본, --vw_source VW_EOM 선택 가능)
    """
    df = df.copy()
    # eom 형식 보정
    if "eom" in df.columns:
        df["eom"] = pd.to_datetime(df["eom"], errors="coerce")

    # EW
    if "country_ret_ew" not in df.columns and "EW" in df.columns:
        df["country_ret_ew"] = pd.to_numeric(df["EW"], errors="coerce")

    # VW
    vw_source = vw_source.upper()
    cand = ["VW_BOM", "VW_EOM"] if vw_source == "VW_BOM" else ["VW_EOM", "VW_BOM"]
    if "country_ret_vw" not in df.columns:
        for c in cand:
            if c in df.columns:
                df["country_ret_vw"] = pd.to_numeric(df[c], errors="coerce")
                break
    return df

def _excess_kurtosis(x: pd.Series) -> float:
    # pandas.Series.kurt()는 excess(Fisher) 반환
    return float(pd.Series(x).kurt())

def _ac1(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    return float(x.autocorr(lag=1)) if len(x) >= 2 else np.nan

def _compute_stats_from_raw(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
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

    out = (use.groupby(["iso3", "period"], dropna=False)
               .apply(_summ)
               .reset_index())
    out.insert(2, "series", value_col)
    return out

def _load_or_build_stats(vw_source: str):
    # 1) p4 결과가 있으면 그대로 사용
    p4, used = _read_first(P4_ALL)
    if p4 is not None:
        need = {"iso3", "period", "series", "n", "mean", "median", "std", "min", "max", "ac1", "skew", "excess_kurtosis"}
        if need.issubset(p4.columns):
            print(f"[STEP] 문제4 결과 사용: {used}")
            return p4

    # 2) raw에서 재계산
    raw, used = _read_first(P2_RAW)
    if raw is None:
        raise FileNotFoundError("입력 파일을 찾지 못했습니다. p4_stats_all.* 또는 hw1_country_monthly_returns.* 가 필요합니다.")
    print(f"[STEP] 문제4 결과 미발견 → 원시 국가수익률에서 통계 재계산: {used}")

    if "eom" not in raw.columns or "iso3" not in raw.columns:
        raise KeyError("raw 파일에 'eom'과 'iso3'가 필요합니다.")
    raw = _ensure_period(raw)
    raw = _alias_from_raw(raw, vw_source=vw_source)

    frames = []
    if "country_ret_ew" in raw.columns:
        frames.append(_compute_stats_from_raw(raw, "country_ret_ew"))
    if "country_ret_vw" in raw.columns:
        frames.append(_compute_stats_from_raw(raw, "country_ret_vw"))
    if not frames:
        raise KeyError("raw 파일에서 사용할 수 있는 수익률 컬럼을 찾지 못했습니다. (EW, VW_BOM, VW_EOM 중 하나 필요)")
    p4_built = pd.concat(frames, ignore_index=True)
    return p4_built

def _attach_group_flag(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    df["group"] = np.where(df["iso3"].isin(DEVELOPED), "Developed",
                    np.where(df["iso3"].isin(EMERGING),  "Emerging", "Other"))
    # 과제는 10개국만 분석
    return df[df["group"].isin(["Developed", "Emerging"])].reset_index(drop=True)

def _group_average(df: pd.DataFrame) -> pd.DataFrame:
    # 그룹 × 기간 × 시리즈별 통계치 평균(국가 평균의 평균)
    cols = ["n", "mean", "median", "std", "min", "max", "ac1", "skew", "excess_kurtosis"]
    out = (df.groupby(["group", "period", "series"], dropna=False)[cols]
             .mean(numeric_only=True)
             .reset_index())
    return out

def _make_pivot(gavg: pd.DataFrame) -> pd.DataFrame:
    metrics = ["n","mean","median","std","min","max","ac1","skew","excess_kurtosis"]
    wide = (gavg
            .set_index(["group","period","series"])
            [metrics]
            .sort_index())
    return wide

def _compare_emerging_minus_dev(gavg: pd.DataFrame) -> pd.DataFrame:
    cols = ["n","mean","median","std","min","max","ac1","skew","excess_kurtosis"]
    dev = gavg[gavg["group"]=="Developed"].set_index(["period","series"])
    eme = gavg[gavg["group"]=="Emerging"].set_index(["period","series"])
    # align 보호
    dev, eme = dev.align(eme, join="inner", axis=0)
    diff = (eme[cols] - dev[cols]).reset_index()
    diff.insert(0, "comparison", "Emerging - Developed")
    return diff

def _changes_between_periods(gavg: pd.DataFrame) -> pd.DataFrame:
    # 각 그룹×시리즈에서 (Recovery - Crisis)
    cols = ["n","mean","median","std","min","max","ac1","skew","excess_kurtosis"]
    cris = gavg[gavg["period"]=="COVID Crisis"].set_index(["group","series"])
    rec  = gavg[gavg["period"]=="Post-crisis Recovery"].set_index(["group","series"])
    cris, rec = cris.align(rec, join="inner", axis=0)
    chg = (rec[cols] - cris[cols]).reset_index()
    chg.insert(0, "period_change", "Recovery - Crisis")
    return chg

def _changes_diff_between_groups(chg: pd.DataFrame) -> pd.DataFrame:
    # 변화의 그룹간 차이: (Emerging 변화) - (Developed 변화)
    cols = ["n","mean","median","std","min","max","ac1","skew","excess_kurtosis"]
    dev = chg[chg["group"]=="Developed"].set_index(["series"])
    eme = chg[chg["group"]=="Emerging"].set_index(["series"])
    dev, eme = dev.align(eme, join="inner", axis=0)
    diff = (eme[cols] - dev[cols]).reset_index()
    diff.insert(0, "comparison", "Δ(EME) - Δ(DEV)")
    return diff

def _fmt(x):
    try:
        return f"{x:,.4f}"
    except Exception:
        return str(x)

# ========= 메인 =========

def main():
    parser = argparse.ArgumentParser(description="Problem 5: Developed vs Emerging 비교분석 (총수정본)")
    parser.add_argument("--vw_source", type=str, default="VW_BOM", choices=["VW_BOM", "VW_EOM"],
                        help="VW 시리즈 소스 선택 (p4 결과 없을 때 raw 재계산용, 기본: VW_BOM)")
    parser.add_argument("--overall", action="store_true",
                        help="기간 구분을 무시하고 전체 표본으로 Developed vs Emerging 비교")
    args = parser.parse_args()

    # 0) 로드 or 빌드
    stats = _load_or_build_stats(vw_source=args.vw_source)

    # stats 형식 점검: series 값 표준화(가능한 경우)
    # 허용 series: 'country_ret_ew', 'country_ret_vw'
    if "series" in stats.columns:
        stats["series"] = stats["series"].replace({
            "EW": "country_ret_ew", "VW": "country_ret_vw",
            "VW_BOM": "country_ret_vw", "VW_EOM": "country_ret_vw"
        })

    # 1) 분류 플래그(10개국만)
    stats = _attach_group_flag(stats)

    # 1.5) --overall 이면 기간을 'All'로 묶음
    if args.overall:
        stats = stats.copy()
        stats["period"] = "All"

    # 2) 그룹 평균(기간×시리즈)
    gavg = _group_average(stats)
    gavg_wide = _make_pivot(gavg)

    # 3) 신흥-선진 차이
    diffs = _compare_emerging_minus_dev(gavg)

    # 4) (기본 모드일 때만) 회복 변화 및 변화의 그룹 간 차이
    if args.overall:
        chg = pd.DataFrame(columns=["period_change","group","series","n","mean","median","std","min","max","ac1","skew","excess_kurtosis"])
        chg_diff = pd.DataFrame(columns=["comparison","series","n","mean","median","std","min","max","ac1","skew","excess_kurtosis"])
    else:
        chg  = _changes_between_periods(gavg)
        chg_diff = _changes_diff_between_groups(chg)

    # ===== 터미널 출력 =====
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("\n" + "="*100)
    print("[개요] 입력 통계 샘플 (상위 10행)")
    print("="*100)
    print(stats.head(10).to_string(index=False))

    print("\n" + "="*100)
    print("[결과] (a) 그룹 평균 통계 — group × period × series")
    print("="*100)
    print(gavg.sort_values(["series","period","group"]).to_string(index=False))

    print("\n" + "="*100)
    print("[결과] (a) 피벗 보기 — index: (group, period, series) / columns: metrics")
    print("="*100)
    print(gavg_wide.head(60).to_string())
    if len(gavg_wide) > 60:
        print(f"... ({len(gavg_wide)} 행)")

    print("\n" + "="*100)
    print("[결과] (b) Emerging - Developed (동일 period·series 기준 차이)")
    print("="*100)
    if len(diffs):
        print(diffs.sort_values(["series","period"]).to_string(index=False))
    else:
        print("(표시할 차이행이 없습니다 — 표본/시리즈 확인)")

    def _brief_insight(df: pd.DataFrame, series_label: str):
        if "period" in df.columns:
            periods = df["period"].unique()
        else:
            periods = ["All"]
        for per in sorted(periods):
            row = df[(df.get("period","All")==per) & (df["series"]==series_label)]
            if len(row):
                r = row.iloc[0]
                print(f" - [{series_label} | {per}] 변동성(σ) 차이(E−D): {_fmt(r.get('std', np.nan))}  "
                      f"왜도 차이(E−D): {_fmt(r.get('skew', np.nan))}  "
                      f"초과첨도 차이(E−D): {_fmt(r.get('excess_kurtosis', np.nan))}  "
                      f"최솟값 차이(E−D): {_fmt(r.get('min', np.nan))}")

    print("\n[해석] (b) 핵심 비교 포인트")
    if len(diffs):
        _brief_insight(diffs, "country_ret_ew")
        _brief_insight(diffs, "country_ret_vw")
    else:
        print(" - 비교 가능한 (Emerging - Developed) 행이 없습니다.")

    if not args.overall:
        print("\n" + "="*100)
        print("[결과] (c) 회복기 변화 — (Recovery - Crisis) by group × series")
        print("="*100)
        if len(chg):
            print(chg.sort_values(["series","group"]).to_string(index=False))
        else:
            print("(표시할 변화행이 없습니다)")

        print("\n" + "="*100)
        print("[결과] (c) 변화의 그룹 간 차이 — Δ(EME) - Δ(DEV) by series")
        print("="*100)
        if len(chg_diff):
            print(chg_diff.sort_values(["series"]).to_string(index=False))
        else:
            print("(표시할 변화차 행이 없습니다)")

    # ===== 저장 =====
    OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    gavg.to_csv(OUT_SUMMARY_CSV, index=False)
    gavg.to_parquet(OUT_SUMMARY_PQ, index=False)
    diffs.to_csv(OUT_DIFFS_CSV, index=False)
    diffs.to_parquet(OUT_DIFFS_PQ, index=False)
    chg.to_csv(OUT_CHG_CSV, index=False)
    chg.to_parquet(OUT_CHG_PQ, index=False)

    print("\n[SAVED] 그룹 평균 통계:", OUT_SUMMARY_CSV, "|", OUT_SUMMARY_PQ)
    print("[SAVED] Emerging-Developed 차이:", OUT_DIFFS_CSV, "|", OUT_DIFFS_PQ)
    print("[SAVED] 회복기 변화 및 그룹 간 변화차:", OUT_CHG_CSV, "|", OUT_CHG_PQ)

    print("\n[DONE] Problem 5 완료.")

if __name__ == "__main__":
    main()
