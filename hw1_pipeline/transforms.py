import numpy as np
import pandas as pd
from .config import COUNTRY_NAME_TO_ISO3, DEV_ISO3
from .progress import iter_progress

def asof_merge_monthly(firm_monthly, funda):
    """
    firm_monthly: 월말 행(필수: gvkey, eom)
    funda: 펀더 데이터(필수: datadate; gvkey 있으면 종목별 as-of, 없어도 전역 as-of)
    """
    if funda.empty:
        return firm_monthly
    if "datadate" not in funda.columns:
        # as-of 기준일이 없으면 병합 불가 → 그대로 반환
        return firm_monthly

    f = funda.copy()
    f["datadate"] = pd.to_datetime(f["datadate"])

    out = []
    # gvkey 기준으로 나눠서 진행(펀더에 gvkey가 없으면 전역으로 동일 적용)
    has_funda_gvkey = ("gvkey" in f.columns)
    uniq = firm_monthly["gvkey"].nunique()
    for k, g in iter_progress(firm_monthly.groupby("gvkey", sort=False),
                              desc="[as-of] fundamentals per gvkey", total=uniq):
        m = g[["eom"]].copy().sort_values("eom")

        if has_funda_gvkey:
            fk = f[f["gvkey"] == k].copy()
            if fk.empty:
                out.append(g.copy())
                continue
        else:
            fk = f.copy()

        fk = fk.sort_values("datadate")

        # as-of 병합 (gvkey는 결과에 포함되지 않으므로 drop에서 다루지 않음)
        g2 = pd.merge_asof(
            m,
            fk,
            left_on="eom",
            right_on="datadate",
            direction="backward",
            tolerance=pd.Timedelta(days=400)
        )

        # datadate만 정리하고, eom 키로 원본 g와 결합
        merged = g.merge(
            g2.drop(columns=["datadate"], errors="ignore"),
            on="eom",
            how="left"
        )
        out.append(merged)

    return pd.concat(out, axis=0, ignore_index=True)

def build_marketcap_hints(df_in):
    df = df_in.copy()
    # 1) monthly px * shr
    px_m  = pd.to_numeric(df.get("px"), errors="coerce")
    shr_m = pd.to_numeric(df.get("shr"), errors="coerce")
    df["mktcap_from_monthly"] = px_m * shr_m
    # 2) mkvalt
    df["mktcap_from_mkvalt"] = np.nan
    for alias in ["mkvaltq","mkvalt"]:
        if alias in df.columns:
            df["mktcap_from_mkvalt"] = pd.to_numeric(df[alias], errors="coerce"); break
    # 3) quarterly px * shr
    prc_q_cols = [c for c in ["prccq","prcc_f","prcc"] if c in df.columns]
    shr_q_cols = [c for c in ["cshoq","csho"] if c in df.columns]
    if prc_q_cols and shr_q_cols:
        prc_q = pd.to_numeric(df[prc_q_cols[0]], errors="coerce")
        shr_q = pd.to_numeric(df[shr_q_cols[0]], errors="coerce")
        scale_q = 1e6 if shr_q_cols[0] in {"cshoq","csho"} else 1.0
        df["mktcap_from_qtr"] = prc_q * (shr_q * scale_q)
    else:
        df["mktcap_from_qtr"] = np.nan
    # 4) pick preferred
    df["mktcap_pref"] = np.nan; df["mktcap_source"] = np.nan
    for tag, col in [("monthly_px_shr","mktcap_from_monthly"),
                     ("mkvalt","mktcap_from_mkvalt"),
                     ("qtr_px_shr","mktcap_from_qtr")]:
        use = df["mktcap_pref"].isna() & pd.to_numeric(df[col], errors="coerce").notna()
        df.loc[use, "mktcap_pref"] = pd.to_numeric(df.loc[use, col], errors="coerce")
        df.loc[use, "mktcap_source"] = tag
    # 5) diagnostics
    def safe_ratio(a, b):
        a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce")
        return np.where((a>0) & (b>0), a/b, np.nan)
    df["ratio_pref_vs_monthly"] = safe_ratio(df["mktcap_pref"], df["mktcap_from_monthly"])
    df["ratio_pref_vs_mkvalt"]  = safe_ratio(df["mktcap_pref"], df["mktcap_from_mkvalt"])
    df["ratio_pref_vs_qtr"]     = safe_ratio(df["mktcap_pref"], df["mktcap_from_qtr"])
    return df

def add_labels(df):
    iso3_to_name = {v:k for k,v in COUNTRY_NAME_TO_ISO3.items()}
    df["country"]    = df["iso3"].map(iso3_to_name)
    df["msci_group"] = df["iso3"].apply(lambda x: "Developed" if x in DEV_ISO3 else "Emerging")
    df["year"]  = df["eom"].dt.year
    df["month"] = df["eom"].dt.month
    return df
