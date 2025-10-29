# hw1_p6_correlation_spillover.py
# Problem 6: Correlation and Spillover Effects — (a)(b)(c) 섹션별 터미널 출력 + CSV/Parquet 저장
# - 입력: data/hw1_country_monthly_returns.(csv|parquet)
# - period 미존재 시 문제3 규칙으로 생성
# - 시리즈: EW 고정 + VW는 기본 VW_BOM (옵션 --vw_source VW_EOM)
# - 산출/저장:
#   (a) 각 하위기간별 국가×국가 상관행렬(EW, VW) → data/p6a_corr_<series>_<period>.*
#       + 쌍별 상관 Long form → data/p6a_corr_pairs_<series>_<period>.*
#   (b) 그룹 내/간 평균 상관 (Dev-Dev, Emg-Emg, Cross) 요약 → data/p6b_corr_summary.*
#   (c) Crisis→Recovery 변화(평균 상관 및 그룹별 평균) → data/p6c_corr_changes.*

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import itertools
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
IN_CAND = [DATA_DIR / "hw1_country_monthly_returns.parquet",
           DATA_DIR / "hw1_country_monthly_returns.csv"]

# 과제의 10개국 (MSCI 분류)
DEVELOPED = {"GBR","DEU","JPN","FRA","AUS"}
EMERGING  = {"CHN","IND","BRA","ZAF","TUR"}
ALL10 = sorted(list(DEVELOPED | EMERGING))

CRISIS_START = pd.Timestamp("2020-03-01")
CRISIS_END   = pd.Timestamp("2021-12-31")
RECOV_START  = pd.Timestamp("2022-01-01")
RECOV_END    = pd.Timestamp("2024-12-31")

PERIODS = [
    ("COVID Crisis", (CRISIS_START, CRISIS_END)),
    ("Post-crisis Recovery", (RECOV_START, RECOV_END)),
]

# (b) 요약 저장
OUT_B_SUMMARY_CSV = DATA_DIR / "p6b_corr_summary.csv"
OUT_B_SUMMARY_PQ  = DATA_DIR / "p6b_corr_summary.parquet"
# (c) 변화 저장
OUT_C_CHANGES_CSV = DATA_DIR / "p6c_corr_changes.csv"
OUT_C_CHANGES_PQ  = DATA_DIR / "p6c_corr_changes.parquet"

def _read_input() -> pd.DataFrame:
    for p in IN_CAND:
        if p.exists():
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, parse_dates=["eom"])
            return df
    raise FileNotFoundError("입력 파일을 찾지 못했습니다: data/hw1_country_monthly_returns.(csv|parquet)")

def _ensure_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "period" in df.columns:
        return df
    e = pd.to_datetime(df["eom"], errors="coerce")
    df["period"] = np.select(
        [
            (e >= CRISIS_START) & (e <= CRISIS_END),
            (e >= RECOV_START)  & (e <= RECOV_END),
        ],
        ["COVID Crisis", "Post-crisis Recovery"],
        default="Out of Sample"
    )
    return df

def _alias_series(df: pd.DataFrame, vw_source: str) -> pd.DataFrame:
    df = df.copy()
    # 표준 이름: country_ret_ew, country_ret_vw
    if "country_ret_ew" not in df.columns and "EW" in df.columns:
        df["country_ret_ew"] = pd.to_numeric(df["EW"], errors="coerce")
    vw_source = vw_source.upper()
    order = ["VW_BOM","VW_EOM"] if vw_source == "VW_BOM" else ["VW_EOM","VW_BOM"]
    if "country_ret_vw" not in df.columns:
        for c in order:
            if c in df.columns:
                df["country_ret_vw"] = pd.to_numeric(df[c], errors="coerce")
                break
    return df

def _wide(df: pd.DataFrame, series_col: str, period_name: str, countries: list[str]) -> pd.DataFrame:
    """행: eom, 열: iso3, 값: series (해당 기간만 필터, 국가 리스트 제한)"""
    use = df.loc[df["period"] == period_name, ["eom","iso3",series_col]].copy()
    use = use[use["iso3"].isin(countries)]
    wide = (use.pivot_table(index="eom", columns="iso3", values=series_col, aggfunc="mean")
               .sort_index())
    return wide

def _corr_and_pairs(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """상관행렬과 pairwise long-form(상삼각) 반환"""
    corr = wide.corr(min_periods=3)  # 최소 3개월 동시관측
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            val = corr.loc[a, b]
            pairs.append({"iso3_a": a, "iso3_b": b, "corr": float(val) if pd.notna(val) else np.nan})
    pairs_df = pd.DataFrame(pairs)
    return corr, pairs_df

def _mean_offdiag(corr: pd.DataFrame) -> float:
    if corr.size == 0:
        return np.nan
    mask = ~np.eye(len(corr), dtype=bool)
    vals = corr.values[mask]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else np.nan

def _group_pair_mask(countries: list[str]):
    """쌍별로 그룹 구분을 계산하기 위한 헬퍼"""
    dev = set(DEVELOPED)
    eme = set(EMERGING)
    pair_tags = {}
    for a, b in itertools.combinations(countries, 2):
        if a in dev and b in dev:
            tag = "Dev-Dev"
        elif a in eme and b in eme:
            tag = "Emg-Emg"
        else:
            tag = "Cross"
        pair_tags[(a,b)] = tag
        pair_tags[(b,a)] = tag
    return pair_tags

def _avg_by_group_pairs(corr: pd.DataFrame) -> dict[str, float]:
    cols = list(corr.columns)
    tags = _group_pair_mask(cols)
    vals = {"Dev-Dev": [], "Emg-Emg": [], "Cross": []}
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            v = corr.loc[a, b]
            if pd.notna(v):
                vals[tags[(a,b)]].append(float(v))
    return {k: (float(np.mean(v)) if len(v) else np.nan) for k, v in vals.items()}

def _print_section_header(tag: str, subtitle: str = ""):
    line = f"[Problem 6{tag}] {subtitle}" if subtitle else f"[Problem 6{tag}]"
    print("\n" + "="*110)
    print(line)
    print("="*110)

def _save_matrix_and_pairs(corr: pd.DataFrame, pairs: pd.DataFrame, series_tag: str, period_name: str):
    base = f"p6a_corr_{series_tag}_{period_name.replace(' ', '_')}"
    m_csv = DATA_DIR / f"{base}.csv"
    m_pq  = DATA_DIR / f"{base}.parquet"
    p_csv = DATA_DIR / f"{base}_pairs.csv"
    p_pq  = DATA_DIR / f"{base}_pairs.parquet"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(m_csv)
    try:
        corr.to_parquet(m_pq, index=True)
    except Exception:
        pass
    pairs.to_csv(p_csv, index=False)
    try:
        pairs.to_parquet(p_pq, index=False)
    except Exception:
        pass

    print(f"[SAVED] (a) corr matrix: {m_csv} | {m_pq}")
    print(f"[SAVED] (a) pairs long : {p_csv} | {p_pq}")

def main():
    parser = argparse.ArgumentParser(description="Problem 6 (a)(b)(c): Correlation & Spillover — 섹션별 출력")
    parser.add_argument("--vw_source", type=str, default="VW_BOM", choices=["VW_BOM","VW_EOM"],
                        help="VW 시리즈 소스 선택 (기본 VW_BOM)")
    parser.add_argument("--series", type=str, default="both", choices=["both","EW","VW"],
                        help="분석 시리즈 선택 (both/EW/VW)")
    parser.add_argument("--min_countries", type=int, default=8,
                        help="상관행렬 계산에 필요한 최소 국가 수(열 개수). 기본 8")
    args = parser.parse_args()

    # 0) 입력 로드 및 표준화
    df = _read_input()
    if "eom" not in df.columns or "iso3" not in df.columns:
        raise KeyError("입력에 eom, iso3 컬럼이 필요합니다.")
    df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
    df = _ensure_period(df)
    df = _alias_series(df, vw_source=args.vw_source)

    # 시리즈 사용 여부
    use_EW = (args.series in ["both","EW"]) and ("country_ret_ew" in df.columns)
    use_VW = (args.series in ["both","VW"]) and ("country_ret_vw" in df.columns)
    if not (use_EW or use_VW):
        raise RuntimeError("사용 가능한 시리즈가 없습니다. (EW 또는 VW_BOM/VW_EOM 중 하나 이상 필요)")

    # 국가 제한: 과제의 10개국 교집합
    avail = sorted([c for c in df["iso3"].unique() if c in ALL10])
    if len(avail) < 2:
        raise RuntimeError("분석 가능한 국가가 충분치 않습니다(과제 10개국 중 최소 2개 필요).")

    print(f"[INFO] 사용 국가({len(avail)}개): {', '.join(avail)}")

    # 누적 요약 (b) 섹션용
    b_summary_rows = []

    # ========= (a) 각 기간별 상관행렬 =========
    def section_a(series_name: str, col: str):
        for period_name, _ in PERIODS:
            _print_section_header("(a)", f"{series_name} — {period_name}: 국가×국가 상관행렬")
            wide = _wide(df, col, period_name, avail)
            # 유효 관측(해당 기간 내 >=3개월) 열만 유지
            valid_cols = [c for c in wide.columns if wide[c].dropna().shape[0] >= 3]
            wide = wide[valid_cols]
            n_c = len(wide.columns)

            print(f"[INFO] 관측월={wide.shape[0]} | 국가수(열)={n_c}")
            if n_c < args.min_countries:
                print(f"[경고] 국가 수가 기준({args.min_countries}) 미만 → 계산은 진행합니다.")

            corr, pairs = _corr_and_pairs(wide)

            print("\n[상관행렬] (전체 출력)")
            with pd.option_context("display.max_columns", 60, "display.width", 180, "display.float_format", "{:,.3f}".format):
                print(corr.to_string())

            # 저장 (a)
            _save_matrix_and_pairs(corr, pairs, series_name.replace(" ", ""), period_name)

            # (b) 요약용 통계 누적
            mean_all = _mean_offdiag(corr)
            grp_avgs = _avg_by_group_pairs(corr)
            b_summary_rows.append({
                "series": series_name,
                "period": period_name,
                "countries_used": n_c,
                "months": int(wide.shape[0]),
                "mean_corr_offdiag": mean_all,
                "mean_corr_dev_dev": grp_avgs.get("Dev-Dev", np.nan),
                "mean_corr_emg_emg": grp_avgs.get("Emg-Emg", np.nan),
                "mean_corr_cross"  : grp_avgs.get("Cross", np.nan),
            })

    if use_EW:
        section_a("EW", "country_ret_ew")
    if use_VW:
        section_a(args.vw_source, "country_ret_vw")  # 표시에 VW_BOM/VW_EOM 구분

    # ========= (b) 그룹 내/간 평균 상관 요약 =========
    _print_section_header("(b)", "그룹 내/간 평균 상관 (Dev-Dev, Emg-Emg, Cross)")
    bsum = pd.DataFrame(b_summary_rows)
    if len(bsum):
        for s in bsum["series"].unique():
            sub = bsum[bsum["series"]==s].copy()
            print(f"\n[Series: {s}] 요약표")
            with pd.option_context("display.float_format", "{:,.3f}".format):
                print(sub[["period","countries_used","months","mean_corr_offdiag",
                           "mean_corr_dev_dev","mean_corr_emg_emg","mean_corr_cross"]]
                      .sort_values("period")
                      .to_string(index=False))

        # 저장 (b)
        OUT_B_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        bsum.to_csv(OUT_B_SUMMARY_CSV, index=False)
        try:
            bsum.to_parquet(OUT_B_SUMMARY_PQ, index=False)
        except Exception:
            pass
        print(f"\n[SAVED] (b) summary: {OUT_B_SUMMARY_CSV} | {OUT_B_SUMMARY_PQ}")
    else:
        print("(표시할 요약이 없습니다 — (a) 단계 결과 없음)")

    # ========= (c) Crisis ↔ Recovery 변화 =========
    _print_section_header("(c)", "Crisis → Recovery 변화(평균 상관 및 그룹별 평균)")
    if len(bsum):
        chg_rows = []
        for s in bsum["series"].unique():
            a = bsum[(bsum["series"]==s) & (bsum["period"]=="COVID Crisis")].reset_index(drop=True)
            b = bsum[(bsum["series"]==s) & (bsum["period"]=="Post-crisis Recovery")].reset_index(drop=True)
            if len(a)==1 and len(b)==1:
                chg_rows.append({
                    "series": s,
                    "d_mean_corr_all": float(b.loc[0,"mean_corr_offdiag"] - a.loc[0,"mean_corr_offdiag"]),
                    "d_dev_dev": float(b.loc[0,"mean_corr_dev_dev"] - a.loc[0,"mean_corr_dev_dev"]),
                    "d_emg_emg": float(b.loc[0,"mean_corr_emg_emg"] - a.loc[0,"mean_corr_emg_emg"]),
                    "d_cross"  : float(b.loc[0,"mean_corr_cross"]   - a.loc[0,"mean_corr_cross"]),
                })
        chg = pd.DataFrame(chg_rows)
        if len(chg):
            print("\n[요약표] 평균 상관 변화 (Recovery - Crisis)")
            with pd.option_context("display.float_format", "{:,.3f}".format):
                print(chg.to_string(index=False))

            # 저장 (c)
            chg.to_csv(OUT_C_CHANGES_CSV, index=False)
            try:
                chg.to_parquet(OUT_C_CHANGES_PQ, index=False)
            except Exception:
                pass
            print(f"\n[SAVED] (c) changes: {OUT_C_CHANGES_CSV} | {OUT_C_CHANGES_PQ}")
        else:
            print("(변화를 계산할 충분한 요약행이 없습니다.)")
    else:
        print("(요약데이터가 없습니다.)")

    print("\n[DONE] Problem 6 (a)(b)(c) 완료.")

if __name__ == "__main__":
    main()
