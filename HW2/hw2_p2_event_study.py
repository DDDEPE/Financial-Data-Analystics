# hw2_p2_event_study.py
# ------------------------------------------------------------
# HW2 Problem 2: Event Study Design
#
# - 목표:
#   1) 글로벌 시장모형 R_it = alpha_i + beta_i * R_mt + eps_it 추정
#   2) 예상수익률 E[R_it] 계산 후 Abnormal Return(AR) 산출
#   3) 이벤트 윈도우 (day 0 ~ +4)에 대해
#        * 일별 AR
#        * 누적 CAR(0~τ) for τ=0,1,2,3,4
#      계산
#
# - 입력:
#   * out/event_panel_day0_to_p4.parquet (또는 CSV fallback)
#   * dwcountryreturns.csv
#   * FTSE All-World Index.csv
#
# - 출력:
#   * out/event_panel_with_ar_car.parquet (long: day별 AR & CAR)
#   * out/event_car_wide.parquet        (wide: CAR0~CAR4 열)
# ------------------------------------------------------------

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# -------------------------
# 설정
# -------------------------
START_DATE = "1995-01-01"
END_DATE   = "2024-12-31"

IN_FILE_RETURNS = "dwcountryreturns.csv"
IN_FILE_WORLD   = "FTSE All-World Index.csv"

OUT_DIR = "out"
PANEL_IN_PARQUET = os.path.join(OUT_DIR, "event_panel_day0_to_p4.parquet")
PANEL_IN_CSV     = os.path.join(OUT_DIR, "event_panel_day0_to_p4.csv")

OUT_PANEL_AR_CAR = os.path.join(OUT_DIR, "event_panel_with_ar_car.parquet")
OUT_CAR_WIDE     = os.path.join(OUT_DIR, "event_car_wide.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    DataFrame에서 후보 이름(candidates) 중 하나와 case-insensitive하게
    매칭되는 컬럼명을 찾아서 돌려줌.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c_lower = cand.lower()
        if c_lower in lower_map:
            return lower_map[c_lower]
    raise ValueError(f"Cannot find any of columns {candidates} in {df.columns.tolist()}")


# ------------------------------------------------------------
# 1. 전체 일별 데이터에서 국가별 시장모형(alpha, beta) 추정
# ------------------------------------------------------------

def load_country_returns(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading country returns from {path} ...")
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("dwcountryreturns.csv 에 'date' 컬럼이 없습니다.")
    df["date_dt"] = pd.to_datetime(df["date"].astype(str))

    # 수익률 컬럼 선택
    if "portret" in df.columns:
        ret_col = "portret"
    elif "portretx" in df.columns:
        ret_col = "portretx"
    else:
        raise ValueError("dwcountryreturns.csv 에서 'portret' 또는 'portretx' 수익률 컬럼을 찾을 수 없습니다.")

    df = df.rename(columns={ret_col: "ret"})

    df = df[(df["date_dt"] >= START_DATE) & (df["date_dt"] <= END_DATE)].copy()
    return df


def load_world_index(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading world index from {path} ...")
    w = pd.read_csv(path)

    date_col = _find_column(w, ["datadate", "date"])
    price_col = _find_column(w, ["prccd", "price", "index_price"])

    w["date_dt"] = pd.to_datetime(w[date_col])
    w = w.sort_values("date_dt").copy()
    w["world_ret"] = w[price_col].pct_change()

    w = w[(w["date_dt"] >= START_DATE) & (w["date_dt"] <= END_DATE)].copy()
    world_ret = w[["date_dt", "world_ret"]].dropna().copy()

    print(f"[INFO] World index sample: {world_ret['date_dt'].min().date()} ~ {world_ret['date_dt'].max().date()}")
    return world_ret


def estimate_market_model(
    returns: pd.DataFrame,
    world: pd.DataFrame,
    min_obs: int = 100
) -> pd.DataFrame:
    """
    국가별로 시장모형 R_it = alpha_i + beta_i * R_mt 추정.
    - 단순 OLS closed-form:
        beta = Cov(R, Rm) / Var(Rm)
        alpha = mean(R) - beta * mean(Rm)
    """

    # 월드 수익률 merge
    df = returns.merge(world, on="date_dt", how="left")
    df = df.dropna(subset=["world_ret"])

    results = []
    for fic, g in df.groupby("fic"):
        g = g.dropna(subset=["ret", "world_ret"])
        n = len(g)
        if n < min_obs:
            print(f"[WARN] {fic}: obs={n} < {min_obs}, alpha=0, beta=1 으로 설정.")
            alpha, beta = 0.0, 1.0
        else:
            r = g["ret"].to_numpy()
            rm = g["world_ret"].to_numpy()

            rm_mean = rm.mean()
            r_mean = r.mean()
            var_rm = ((rm - rm_mean) ** 2).mean()

            if var_rm <= 0:
                print(f"[WARN] {fic}: Var(Rm)=0, alpha=0, beta=1 으로 설정.")
                alpha, beta = 0.0, 1.0
            else:
                cov = ((rm - rm_mean) * (r - r_mean)).mean()
                beta = cov / var_rm
                alpha = r_mean - beta * rm_mean

        results.append({"fic": fic, "alpha": alpha, "beta": beta})

    beta_df = pd.DataFrame(results)
    print("[INFO] Estimated market model parameters (first few):")
    print(beta_df.head())
    return beta_df


# ------------------------------------------------------------
# 2. Problem 1의 이벤트 패널 불러오기
# ------------------------------------------------------------

def load_event_panel() -> pd.DataFrame:
    if os.path.exists(PANEL_IN_PARQUET):
        print(f"[INFO] Loading event panel from {PANEL_IN_PARQUET}")
        panel = pd.read_parquet(PANEL_IN_PARQUET)
    elif os.path.exists(PANEL_IN_CSV):
        print(f"[INFO] Loading event panel from {PANEL_IN_CSV}")
        panel = pd.read_csv(PANEL_IN_CSV, parse_dates=["ret_date", "fomc_date"])
    else:
        raise FileNotFoundError(
            f"Event panel file not found: {PANEL_IN_PARQUET} or {PANEL_IN_CSV}. "
            "먼저 hw2_p1_data_construction.py 를 실행해 주세요."
        )

    # 날짜형 보정
    if not np.issubdtype(panel["ret_date"].dtype, np.datetime64):
        panel["ret_date"] = pd.to_datetime(panel["ret_date"])
    if not np.issubdtype(panel["fomc_date"].dtype, np.datetime64):
        panel["fomc_date"] = pd.to_datetime(panel["fomc_date"])

    return panel


# ------------------------------------------------------------
# 3. AR, CAR 계산
# ------------------------------------------------------------

def compute_ar_car(
    panel: pd.DataFrame,
    betas: pd.DataFrame,
    max_horizon: int = 4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - panel: Problem 1에서 생성한 event_panel_day0_to_p4
        (필수 컬럼: fic, fomc_date, event_day, ret, world_ret)
    - betas: 국가별 alpha, beta

    반환:
      1) long form: event_day별 AR, CAR
      2) wide form: CAR0~CAR4 열로 정리
    """

    # alpha, beta 붙이기
    panel = panel.merge(betas, on="fic", how="left")

    if panel["alpha"].isna().any() or panel["beta"].isna().any():
        missing = panel.loc[panel["alpha"].isna() | panel["beta"].isna(), "fic"].unique()
        print(f"[WARN] 다음 국가들은 alpha/beta가 없습니다 (시장모형 추정에 포함 안 됐을 수 있음): {missing}")

    # 예상수익률 & AR 계산
    panel["exp_ret"] = panel["alpha"] + panel["beta"] * panel["world_ret"]
    panel["abret"]   = panel["ret"] - panel["exp_ret"]

    # (fic, fomc_date)별 event_day 정렬 후 CAR 계산
    panel = panel.sort_values(["fic", "fomc_date", "event_day"]).copy()

    def _cum_abret(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("event_day").copy()
        g["car"] = g["abret"].cumsum()
        return g

    panel = panel.groupby(["fic", "fomc_date"], group_keys=False).apply(_cum_abret)

    # long-form: event_day별 AR, CAR 그대로
    panel_long = panel.copy()

    # wide-form: CAR0~CAR4로 pivot
    wide_list = []
    for (fic, fomc_date), g in panel.groupby(["fic", "fomc_date"]):
        g = g.sort_values("event_day")
        # event_day 0~4만 사용
        g = g[g["event_day"].between(0, max_horizon)]

        rec = {
            "fic": fic,
            "fomc_date": fomc_date,
            # 한 번만 들어가면 되는 정보들
            "country": g["country"].iloc[0] if "country" in g.columns else None,
            "currency": g["currency"].iloc[0] if "currency" in g.columns else None,
            "NS": g["NS"].iloc[0],
            "GSS_target": g["GSS_target"].iloc[0],
            "GSS_path": g["GSS_path"].iloc[0],
        }

        for h in range(0, max_horizon + 1):
            sub = g[g["event_day"] == h]
            if len(sub) == 0:
                rec[f"CAR{h}"] = np.nan
            else:
                rec[f"CAR{h}"] = sub["car"].iloc[0]

        wide_list.append(rec)

    car_wide = pd.DataFrame(wide_list)
    car_wide = car_wide.sort_values(["fic", "fomc_date"]).reset_index(drop=True)

    return panel_long, car_wide


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    # 1) 전체 일별 데이터 + 월드 인덱스 로드 & 시장모형 추정
    returns = load_country_returns(IN_FILE_RETURNS)
    world   = load_world_index(IN_FILE_WORLD)
    betas   = estimate_market_model(returns, world, min_obs=100)

    # 2) Problem 1의 이벤트 패널 로드
    panel = load_event_panel()

    # 3) AR & CAR 계산
    panel_long, car_wide = compute_ar_car(panel, betas, max_horizon=4)

    # 4) 저장
    try:
        panel_long.to_parquet(OUT_PANEL_AR_CAR, index=False)
        print(f"[INFO] Saved long-form panel (with AR & CAR) to {OUT_PANEL_AR_CAR}")
    except Exception as e:
        print(f"[WARN] Failed to save {OUT_PANEL_AR_CAR} ({e}), saving CSV instead.")
        panel_long.to_csv(OUT_PANEL_AR_CAR.replace(".parquet", ".csv"), index=False)
        print(f"[INFO] Saved CSV to {OUT_PANEL_AR_CAR.replace('.parquet', '.csv')}")

    try:
        car_wide.to_parquet(OUT_CAR_WIDE, index=False)
        print(f"[INFO] Saved wide-form CAR to {OUT_CAR_WIDE}")
    except Exception as e:
        print(f"[WARN] Failed to save {OUT_CAR_WIDE} ({e}), saving CSV instead.")
        car_wide.to_csv(OUT_CAR_WIDE.replace(".parquet", ".csv"), index=False)
        print(f"[INFO] Saved CSV to {OUT_CAR_WIDE.replace('.parquet', '.csv')}")

    print("[DONE] Problem 2 event study construction complete.")


if __name__ == "__main__":
    main()
