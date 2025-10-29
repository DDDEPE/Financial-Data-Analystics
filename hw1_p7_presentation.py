# hw1_p7_presentation.py  (v2)
# Problem 7: Presentation of Results (a)(b)(c)
# - (b) 히스토그램: 기본값을 "전체 국가"로 변경 (과제 10개국 전부 × 2기간 × 2시리즈)
# - 선택 옵션:
#    --hist_scope {all,reps,dev,emg,list} (기본 all)
#    --iso3_list  "GBR,DEU,..."  (hist_scope=list 일 때 사용)
#    --dev_rep / --emg_rep (hist_scope=reps일 때 대표국 지정)

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
P4_ALL = [DATA_DIR / "p4_stats_all.parquet", DATA_DIR / "p4_stats_all.csv"]
P2_RAW = [DATA_DIR / "hw1_country_monthly_returns.parquet", DATA_DIR / "hw1_country_monthly_returns.csv"]

DEVELOPED = {"GBR","DEU","JPN","FRA","AUS"}
EMERGING  = {"CHN","IND","BRA","ZAF","TUR"}
ALL10 = sorted(list(DEVELOPED | EMERGING))

CRISIS_START = pd.Timestamp("2020-03-01")
CRISIS_END   = pd.Timestamp("2021-12-31")
RECOV_START  = pd.Timestamp("2022-01-01")
RECOV_END    = pd.Timestamp("2024-12-31")

PLOTS_DIR = DATA_DIR / "plots"
OUT_A_CSV = DATA_DIR / "p7_descriptive_table.csv"
OUT_A_PQ  = DATA_DIR / "p7_descriptive_table.parquet"
OUT_C_TXT = DATA_DIR / "p7_discussion.txt"

def _read_first(paths):
    for p in paths:
        if p.exists():
            if p.suffix.lower()==".parquet":
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
    df = df.copy()
    if "eom" in df.columns:
        df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
    if "country_ret_ew" not in df.columns and "EW" in df.columns:
        df["country_ret_ew"] = pd.to_numeric(df["EW"], errors="coerce")
    vw_source = vw_source.upper()
    order = ["VW_BOM","VW_EOM"] if vw_source=="VW_BOM" else ["VW_EOM","VW_BOM"]
    if "country_ret_vw" not in df.columns:
        for c in order:
            if c in df.columns:
                df["country_ret_vw"] = pd.to_numeric(df[c], errors="coerce")
                break
    return df

def _excess_kurtosis(x: pd.Series) -> float:
    return float(pd.Series(x).kurt())

def _ac1(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    return float(x.autocorr(lag=1)) if len(x) >= 2 else np.nan

def _compute_stats_from_raw(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    use = df[["iso3","period","eom",value_col]].copy()
    use[value_col] = pd.to_numeric(use[value_col], errors="coerce")
    use = use.dropna(subset=[value_col])

    def _summ(g: pd.DataFrame) -> pd.Series:
        x = g[value_col]
        n = int(x.notna().sum())
        return pd.Series({
            "n": n, "mean": float(x.mean()) if n else np.nan,
            "median": float(x.median()) if n else np.nan,
            "std": float(x.std(ddof=1)) if n>=2 else np.nan,
            "min": float(x.min()) if n else np.nan,
            "max": float(x.max()) if n else np.nan,
            "ac1": _ac1(x),
            "skew": float(x.skew()) if n>=3 else np.nan,
            "excess_kurtosis": _excess_kurtosis(x) if n>=4 else np.nan,
        })

    out = (use.groupby(["iso3","period"], dropna=False)
               .apply(_summ)
               .reset_index())
    out.insert(2, "series", value_col)
    return out

def _load_stats_or_build(vw_source: str) -> pd.DataFrame:
    p4, used = _read_first(P4_ALL)
    if p4 is not None:
        need = {"iso3","period","series","n","mean","median","std","min","max","ac1","skew","excess_kurtosis"}
        if need.issubset(set(p4.columns)):
            print(f"[STEP] 문제4 결과 사용: {used}")
            p4["series"] = p4["series"].replace({
                "EW": "country_ret_ew", "VW": "country_ret_vw",
                "VW_BOM": "country_ret_vw", "VW_EOM": "country_ret_vw"
            })
            return p4

    raw, used = _read_first(P2_RAW)
    if raw is None:
        raise FileNotFoundError("입력 없음: p4_stats_all.* 또는 hw1_country_monthly_returns.* 필요")
    print(f"[STEP] 문제4 결과 미발견 → raw에서 통계 재계산: {used}")
    if "eom" not in raw.columns or "iso3" not in raw.columns:
        raise KeyError("raw 파일에 'eom'과 'iso3' 필요")
    raw = _ensure_period(raw)
    raw = _alias_from_raw(raw, vw_source=vw_source)

    frames = []
    if "country_ret_ew" in raw.columns:
        frames.append(_compute_stats_from_raw(raw, "country_ret_ew"))
    if "country_ret_vw" in raw.columns:
        frames.append(_compute_stats_from_raw(raw, "country_ret_vw"))
    if not frames:
        raise KeyError("raw에서 사용할 수익률 컬럼 없음(EW, VW_BOM, VW_EOM 중 하나 필요)")
    return pd.concat(frames, ignore_index=True)

def _attach_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group"] = np.where(out["iso3"].isin(DEVELOPED), "Developed",
                     np.where(out["iso3"].isin(EMERGING), "Emerging", "Other"))
    return out[out["group"].isin(["Developed","Emerging"])].reset_index(drop=True)

def _make_a_table(stats: pd.DataFrame) -> pd.DataFrame:
    cols = ["iso3","group","period","series","n","mean","median","std","min","max","ac1","skew","excess_kurtosis"]
    a_tbl = stats[cols].sort_values(["group","iso3","series","period"]).reset_index(drop=True)
    return a_tbl

def _build_raw_for_plots(vw_source: str) -> pd.DataFrame:
    raw, used = _read_first(P2_RAW)
    if raw is None:
        raise FileNotFoundError("히스토그램 생성을 위해 hw1_country_monthly_returns.* 가 필요합니다.")
    print(f"[STEP] 히스토그램용 raw 사용: {used}")
    raw = _ensure_period(raw)
    raw = _alias_from_raw(raw, vw_source=vw_source)
    return raw

def _plot_histograms_for_country(raw: pd.DataFrame, iso3: str, vw_source: str, bins: int = 20):
    """해당 국가 1개: 2기간 × 2시리즈(가능한 것만) 히스토그램 저장"""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    periods = ["COVID Crisis","Post-crisis Recovery"]
    series_map = {"EW":"country_ret_ew", vw_source: "country_ret_vw"}

    for s_label, col in series_map.items():
        if col not in raw.columns:
            continue
        for per in periods:
            sub = raw[(raw["iso3"]==iso3) & (raw["period"]==per)]
            x = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(x)==0:
                continue
            plt.figure(figsize=(6,4))
            plt.hist(x, bins=bins, edgecolor="black")
            plt.title(f"{iso3} — {s_label} — {per}")
            plt.xlabel("Monthly Return")
            plt.ylabel("Frequency")
            out_png = PLOTS_DIR / f"p7_hist_{iso3}_{s_label.replace(' ','')}_{per.replace(' ','_')}.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"[SAVED] (b) histogram: {out_png}")

def _choose_reps(raw: pd.DataFrame, dev_rep: str|None, emg_rep: str|None) -> tuple[str,str]:
    devs = sorted([c for c in raw["iso3"].unique() if c in DEVELOPED])
    emgs = sorted([c for c in raw["iso3"].unique() if c in EMERGING])
    d = dev_rep if dev_rep in devs else (devs[0] if devs else None)
    e = emg_rep if emg_rep in emgs else (emgs[0] if emgs else None)
    if d is None or e is None:
        raise RuntimeError("대표 국가를 선택할 수 없습니다(데이터 내 선진/신흥 국가 부족).")
    return d, e

def _parse_iso3_list(txt: str) -> list[str]:
    return [s.strip().upper() for s in txt.split(",") if s.strip()]

def _auto_discussion(a_tbl: pd.DataFrame) -> str:
    m = (a_tbl.groupby(["group","period","series"])
               [["mean","std","skew","excess_kurtosis","min","max"]]
               .mean(numeric_only=True)
               .reset_index())
    def _val(g, per, ser, col):
        try:
            return float(m[(m["group"]==g)&(m["period"]==per)&(m["series"]==ser)][col].iloc[0])
        except Exception:
            return np.nan

    ser_EW = "country_ret_ew"; ser_VW = "country_ret_vw"
    lines = []

    d_std_c_ew = _val("Developed","COVID Crisis",ser_EW,"std")
    e_std_c_ew = _val("Emerging","COVID Crisis",ser_EW,"std")
    d_mean_c_ew= _val("Developed","COVID Crisis",ser_EW,"mean")
    e_mean_c_ew= _val("Emerging","COVID Crisis",ser_EW,"mean")
    d_min_c_ew = _val("Developed","COVID Crisis",ser_EW,"min")
    e_min_c_ew = _val("Emerging","COVID Crisis",ser_EW,"min")

    lines.append(
        f"During the COVID-19 crisis, emerging markets showed "
        f"{'higher' if (e_std_c_ew>d_std_c_ew) else 'lower' if (e_std_c_ew<d_std_c_ew) else 'similar'} "
        f"volatility than developed markets in EW returns (σ_EME={e_std_c_ew:.3f} vs σ_DEV={d_std_c_ew:.3f}). "
        f"Average returns were "
        f"{'higher' if e_mean_c_ew>d_mean_c_ew else 'lower' if e_mean_c_ew<d_mean_c_ew else 'similar'}, "
        f"and worst months were "
        f"{'more severe' if e_min_c_ew<d_min_c_ew else 'less severe' if e_min_c_ew>d_min_c_ew else 'comparable'} "
        f"in emerging markets (min_EME={e_min_c_ew:.3f}, min_DEV={d_min_c_ew:.3f})."
    )

    d_std_r_ew = _val("Developed","Post-crisis Recovery",ser_EW,"std")
    e_std_r_ew = _val("Emerging","Post-crisis Recovery",ser_EW,"std")
    d_mean_r_ew= _val("Developed","Post-crisis Recovery",ser_EW,"mean")
    e_mean_r_ew= _val("Emerging","Post-crisis Recovery",ser_EW,"mean")
    d_delta_std = d_std_r_ew - d_std_c_ew
    e_delta_std = e_std_r_ew - e_std_c_ew

    lines.append(
        f"In the recovery, volatility declined by {d_delta_std:.3f} (DEV) and {e_delta_std:.3f} (EME) in EW terms, "
        f"with averages at DEV={d_mean_r_ew:.3f} and EME={e_mean_r_ew:.3f}. "
        f"This indicates "
        f"{'faster stabilization in emerging' if e_delta_std<d_delta_std else 'faster stabilization in developed' if d_delta_std<e_delta_std else 'a similar pace of stabilization'}."
    )

    d_std_c_vw = _val("Developed","COVID Crisis",ser_VW,"std")
    e_std_c_vw = _val("Emerging","COVID Crisis",ser_VW,"std")
    d_std_r_vw = _val("Developed","Post-crisis Recovery",ser_VW,"std")
    e_std_r_vw = _val("Emerging","Post-crisis Recovery",ser_VW,"std")
    if np.isfinite(d_std_c_vw) and np.isfinite(e_std_c_vw):
        lines.append(
            f"Using VW returns, crisis volatility (σ_DEV={d_std_c_vw:.3f}, σ_EME={e_std_c_vw:.3f}) converged in recovery "
            f"to σ_DEV={d_std_r_vw:.3f} and σ_EME={e_std_r_vw:.3f}, consistent with normalization of risk premia."
        )

    return "\n\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Problem 7: Presentation of Results (a)(b)(c) — v2")
    parser.add_argument("--vw_source", type=str, default="VW_BOM", choices=["VW_BOM","VW_EOM"],
                        help="VW 시리즈 선택(히스토그램/재계산용)")
    parser.add_argument("--bins", type=int, default=20, help="히스토그램 bins (기본 20)")
    parser.add_argument("--hist_scope", type=str, default="all",
                        choices=["all","reps","dev","emg","list"],
                        help="히스토그램 대상 선택: all(기본)/reps/dev/emg/list")
    parser.add_argument("--iso3_list", type=str, default="",
                        help="hist_scope=list 일 때 ISO3 목록(콤마 구분). 예: \"GBR,DEU,JPN\"")
    parser.add_argument("--dev_rep", type=str, default="GBR", help="hist_scope=reps 일 때 대표 선진국 ISO3")
    parser.add_argument("--emg_rep", type=str, default="IND", help="hist_scope=reps 일 때 대표 신흥국 ISO3")
    args = parser.parse_args()

    # (a) 통계 로드/생성
    stats = _load_stats_or_build(vw_source=args.vw_source)
    stats = stats[stats["iso3"].isin(ALL10)].copy()
    stats = _attach_group(stats)
    a_tbl = _make_a_table(stats)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("\n" + "="*100)
    print("[Problem 7(a)] 국가·기간·그룹별 요약통계 (상위 30행 미리보기)")
    print("="*100)
    print(a_tbl.head(30).to_string(index=False))

    OUT_A_CSV.parent.mkdir(parents=True, exist_ok=True)
    a_tbl.to_csv(OUT_A_CSV, index=False)
    try:
        a_tbl.to_parquet(OUT_A_PQ, index=False)
    except Exception:
        pass
    print(f"\n[SAVED] (a) descriptive table: {OUT_A_CSV} | {OUT_A_PQ}")

    # (b) 히스토그램: 대상 국가 선정
    raw = _build_raw_for_plots(vw_source=args.vw_source)
    raw = raw[raw["iso3"].isin(ALL10)].copy()

    if args.hist_scope == "all":
        countries = sorted(raw["iso3"].unique())  # 과제 10개국 모두 (데이터 내 존재하는 것)
    elif args.hist_scope == "dev":
        countries = sorted([c for c in raw["iso3"].unique() if c in DEVELOPED])
    elif args.hist_scope == "emg":
        countries = sorted([c for c in raw["iso3"].unique() if c in EMERGING])
    elif args.hist_scope == "list":
        wanted = set(_parse_iso3_list(args.iso3_list))
        countries = sorted([c for c in raw["iso3"].unique() if c in wanted])
        if not countries:
            raise RuntimeError("iso3_list에 해당하는 국가가 데이터에 없습니다.")
    else:  # reps
        d, e = _choose_reps(raw, args.dev_rep, args.emg_rep)
        countries = [d, e]

    print("\n" + "="*100)
    print(f"[Problem 7(b)] 히스토그램 생성 — 대상 국가({len(countries)}개): {', '.join(countries)}")
    print("="*100)

    total_expected = 0
    for iso3 in countries:
        _plot_histograms_for_country(raw, iso3, args.vw_source, bins=args.bins)
        # 최대 4장(2기간×2시리즈) 기준으로 예상 갯수 합산(실제는 결측으로 줄 수 있음)
        total_expected += 4
    print(f"\n[INFO] 히스토그램 예상 최대 {total_expected}장 (결측 여부에 따라 일부 건너뜀)")

    # (c) 자동 요약
    print("\n" + "="*100)
    print("[Problem 7(c)] 자동 요약(2–3 문단)")
    print("="*100)
    discussion = _auto_discussion(a_tbl)
    print(discussion if discussion.strip() else "(요약을 생성할 충분한 통계가 없습니다.)")

    with open(OUT_C_TXT, "w", encoding="utf-8") as f:
        f.write(discussion)
    print(f"\n[SAVED] (c) discussion: {OUT_C_TXT}")

    print("\n[DONE] Problem 7 완료 (v2).")

if __name__ == "__main__":
    main()
