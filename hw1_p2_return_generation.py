# hw1_p2_return_generation.py
# 문제 2 (a)(b)(c) 계산 + CSV 저장 + 콘솔 요약 출력

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
BASE_CANDIDATES = [
    DATA_DIR / "hw1_firm_monthly_base.parquet",  # 있으면 읽기만
    DATA_DIR / "hw1_firm_monthly_base.csv",
]

OUT_FIRM_ENH_CSV = DATA_DIR / "hw1_firm_monthly_enhanced.csv"
OUT_COUNTRY_CSV  = DATA_DIR / "hw1_country_monthly_returns.csv"

def _load_base():
    for p in BASE_CANDIDATES:
        if p.exists():
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, parse_dates=["eom"])
            return df
    raise FileNotFoundError("data/hw1_firm_monthly_base.(parquet|csv)를 찾을 수 없습니다. 먼저 1번 스크립트를 실행하세요.")

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "eom" not in df.columns:
        raise ValueError("필수 열 eom 이 없습니다.")
    df["eom"] = pd.to_datetime(df["eom"])
    # 없을 수 있는 열들 보강
    for c in ["gvkey","iid","iso3","country","px","shr","mktcap","mret"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _entity_keys(df: pd.DataFrame):
    """실제 값이 존재하는 식별자만 키로 사용 (iid가 전부 NaN이면 제외)"""
    keys = ["gvkey"]
    if "iid" in df.columns and df["iid"].notna().any():
        keys.append("iid")
    return keys

def _rebuild_mret_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    # (a) mret 점검/재구성
    if df["mret"].notna().sum() > 0:
        return df
    if df["px"].notna().sum() == 0:
        raise RuntimeError("mret가 전부 결측이고 px 도 없어 재구성 불가합니다.")
    key = _entity_keys(df)
    df = df.sort_values(key + ["eom"]).copy()
    px = pd.to_numeric(df["px"], errors="coerce")
    lag = df.groupby(key)["px"].shift(1)
    df["mret"] = np.where((lag.notna()) & (lag != 0), px/lag - 1.0, np.nan)
    return df

def _reconstruct_mktcap(df: pd.DataFrame) -> pd.DataFrame:
    """ (b) mktcap 결측 보강: 우선순위 (원본 mktcap) -> (px*shr) [필요 시 ×1e6 스케일] """
    df = df.copy()
    df["mktcap"] = pd.to_numeric(df.get("mktcap"), errors="coerce")
    df["px"]     = pd.to_numeric(df.get("px"),     errors="coerce")
    df["shr"]    = pd.to_numeric(df.get("shr"),    errors="coerce")

    # 기본 보조값: px * shr
    df["mkt_from_px_shr"] = df["px"] * df["shr"]

    key = _entity_keys(df)

    # 그룹(기업)별 중앙값 요약
    g = (df.groupby(key, dropna=False)
           .agg(base_med=("mktcap", "median"),
                med_raw =("mkt_from_px_shr", "median"),
                n_base  =("mktcap", lambda s: int(s.notna().sum())))
           .reset_index())

    # 1e6 스케일 적용 여부 결정(결측은 False로 처리)
    med_scaled   = g["med_raw"] * 1_000_000.0
    # base가 충분할 때: base_med에 더 가까운 쪽 선택
    near_scaled  = (g["base_med"] - med_scaled).abs()
    near_raw     = (g["base_med"] - g["med_raw"]).abs()
    choose_scaled_pref = (near_scaled < near_raw)

    cond_has_base = g["n_base"].ge(6) & g["base_med"].notna()
    rule_base     = cond_has_base & choose_scaled_pref.fillna(False)

    # base가 부족할 때: 규모 휴리스틱
    rule_fallback = (~cond_has_base) & (
        g["med_raw"].gt(0) &
        g["med_raw"].lt(1e3) &
        med_scaled.between(1e6, 1e13, inclusive="both")
    )

    use_scaled = (rule_base | rule_fallback).fillna(False)
    # ⚠️ NAType.__bool__ 오류 방지: .to_numpy()로 넘겨서 np.where에 전달
    g["scale_factor"] = np.where(use_scaled.to_numpy(), 1_000_000.0, 1.0)

    # 스케일 머지 후 최종 EOM 시총 산출
    df = df.merge(g[key + ["scale_factor"]], on=key, how="left")
    df["scale_factor"] = df["scale_factor"].fillna(1.0)

    from_px_shr_scaled = df["mkt_from_px_shr"] * df["scale_factor"]
    df["mktcap_eom"]   = np.where(df["mktcap"].notna(), df["mktcap"], from_px_shr_scaled)

    # 0/음수 제거
    df["mktcap_eom"] = np.where(pd.to_numeric(df["mktcap_eom"], errors="coerce") > 0,
                                df["mktcap_eom"], np.nan)
    return df

def _build_mktcap_bom(df: pd.DataFrame) -> pd.DataFrame:
    # BOM = 직전월말
    key = _entity_keys(df)
    df = df.sort_values(key + ["eom"]).copy()
    df["mktcap_bom"] = df.groupby(key)["mktcap_eom"].shift(1)
    df["mktcap_bom"] = np.where(df["mktcap_bom"] > 0, df["mktcap_bom"], np.nan)
    return df

def _country_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    # (c) 국가별 EW, VW-BOM, VW-EOM
    base = df[["country","iso3","eom","gvkey","iid","mret","mktcap_bom","mktcap_eom"]].copy()
    base["mret"] = pd.to_numeric(base["mret"], errors="coerce")

    # EW
    ew = (base.dropna(subset=["mret"])
               .groupby(["country","iso3","eom"], as_index=False)
               .agg(ew_ret=("mret","mean"),
                    n_firms=("gvkey","nunique")))

    # VW-BOM
    bom = base.dropna(subset=["mret","mktcap_bom"]).copy()
    if not bom.empty:
        bom["w_sum"] = bom.groupby(["country","iso3","eom"])["mktcap_bom"].transform("sum")
        bom["w"]     = np.where(bom["w_sum"]>0, bom["mktcap_bom"]/bom["w_sum"], np.nan)
        vw_bom = (bom.dropna(subset=["w"])
                     .assign(vw_bom=lambda x: x["w"]*x["mret"])
                     .groupby(["country","iso3","eom"], as_index=False)
                     .agg(vw_bom=("vw_bom","sum"),
                          n_firms_bom=("gvkey","nunique")))
    else:
        vw_bom = ew[["country","iso3","eom"]].copy()
        vw_bom["vw_bom"] = np.nan
        vw_bom["n_firms_bom"] = 0

    # VW-EOM
    eom = base.dropna(subset=["mret","mktcap_eom"]).copy()
    if not eom.empty:
        eom["w_sum"] = eom.groupby(["country","iso3","eom"])["mktcap_eom"].transform("sum")
        eom["w"]     = np.where(eom["w_sum"]>0, eom["mktcap_eom"]/eom["w_sum"], np.nan)
        vw_eom = (eom.dropna(subset=["w"])
                     .assign(vw_eom=lambda x: x["w"]*x["mret"])
                     .groupby(["country","iso3","eom"], as_index=False)
                     .agg(vw_eom=("vw_eom","sum"),
                          n_firms_eom=("gvkey","nunique")))
    else:
        vw_eom = ew[["country","iso3","eom"]].copy()
        vw_eom["vw_eom"] = np.nan
        vw_eom["n_firms_eom"] = 0

    out = (ew.merge(vw_bom, on=["country","iso3","eom"], how="outer")
             .merge(vw_eom, on=["country","iso3","eom"], how="outer")
             .sort_values(["country","eom"])
             .reset_index(drop=True))

    # 커버리지 리포트
    cov = (out.groupby("country", as_index=False)
              .agg(ew_obs=("ew_ret", lambda s: int(s.notna().sum())),
                   vw_bom_obs=("vw_bom", lambda s: int(s.notna().sum())),
                   vw_eom_obs=("vw_eom", lambda s: int(s.notna().sum()))))
    print("\n=== Coverage by country (obs counts) ===")
    print(cov.to_string(index=False))

    return out

def _save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved CSV:", path.resolve())

def _pretty_print_abc(firm: pd.DataFrame, country_m: pd.DataFrame):
    # (A) Firm-level monthly returns
    print("\n================ (A) Firm-level Monthly Returns (sample) ================")
    cols_a = ["gvkey","iid","country","iso3","eom","px","mret"]
    cols_a = [c for c in cols_a if c in firm.columns]
    print(firm[cols_a].sort_values(["gvkey","iid","eom"]).head(12).to_string(index=False))

    # (B) Firm-level market cap (EOM & BOM)
    print("\n================ (B) Firm-level Market Cap EOM/BOM (sample) ================")
    cols_b = ["gvkey","iid","country","iso3","eom","mktcap","mktcap_eom","mktcap_bom","scale_factor"]
    cols_b = [c for c in cols_b if c in firm.columns]
    print(firm[cols_b].sort_values(["gvkey","iid","eom"]).head(12).to_string(index=False))

    # (C) Country-level EW/VW returns
    print("\n================ (C) Country-level EW / VW Returns (sample) ================")
    cols_c = ["country","iso3","eom","ew_ret","vw_bom","vw_eom","n_firms","n_firms_bom","n_firms_eom"]
    cols_c = [c for c in cols_c if c in country_m.columns]
    print(country_m[cols_c].sort_values(["country","eom"]).head(20).to_string(index=False))

def main():
    firm = _load_base()
    firm = _ensure_cols(firm)

    # (a) mret 보정
    firm = _rebuild_mret_if_needed(firm)

    # (b) mktcap 보강(EOM) + BOM 생성
    firm = _reconstruct_mktcap(firm)
    firm = _build_mktcap_bom(firm)

    # 보강본 저장 (CSV)
    _save_csv(firm, OUT_FIRM_ENH_CSV)

    # (c) 국가별 월별 수익률
    country_m = _country_monthly_panel(firm)
    _save_csv(country_m, OUT_COUNTRY_CSV)

    # === 콘솔 요약 (A)(B)(C) 한 번에 출력 ===
    _pretty_print_abc(firm, country_m)

if __name__ == "__main__":
    main()
