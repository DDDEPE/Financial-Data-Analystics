# hw2_p3_regression.py
# ------------------------------------------------------------
# HW2 Problem 3: Regression Analysis on Monetary Policy Shocks
#
# 목표:
#   - Problem 2에서 계산한 CAR_h (h = 0,1,2,3,4)를 사용해서
#       R_i,t^h = β1 * NS_t + μ_i + λ_t + ε_i,t
#     패널 회귀를 추정.
#   - 여기서:
#       * R_i,t^h : 국가 i, 이벤트 t에 대한 이벤트 윈도우 누적수익(CAR_h)
#       * NS_t    : t 시점 FOMC 회의의 NS shock
#       * μ_i     : 국가 고정효과 (country fixed effects)
#       * λ_t     : 이벤트 날짜 고정효과 (event-date fixed effects)
#
# 입력:
#   - out/event_car_wide.parquet  (또는 CSV fallback)
#     (hw2_p2_event_study.py에서 생성)
#
# 출력:
#   - out/regression_results_ns_by_horizon.csv
#       : horizon별 β1, 표준오차, t값, p값, 관측치 수, R^2
#   - out/regression_summary_h{h}.txt
#       : 각 horizon별 회귀 summary 텍스트
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from patsy import dmatrices

# -------------------------
# 설정
# -------------------------
OUT_DIR = "out"
CAR_WIDE_IN_PARQUET = os.path.join(OUT_DIR, "event_car_wide.parquet")
CAR_WIDE_IN_CSV     = os.path.join(OUT_DIR, "event_car_wide.csv")

OUT_RESULT_CSV = os.path.join(OUT_DIR, "regression_results_ns_by_horizon.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. CAR 데이터 로드
# ------------------------------------------------------------

def load_car_wide() -> pd.DataFrame:
    """
    hw2_p2_event_study.py에서 생성한 event_car_wide 파일 로드.
    필수 컬럼:
      - fic, fomc_date, NS
      - CAR0, CAR1, CAR2, CAR3, CAR4
    """
    if os.path.exists(CAR_WIDE_IN_PARQUET):
        print(f"[INFO] Loading CAR data from {CAR_WIDE_IN_PARQUET}")
        df = pd.read_parquet(CAR_WIDE_IN_PARQUET)
    elif os.path.exists(CAR_WIDE_IN_CSV):
        print(f"[INFO] Loading CAR data from {CAR_WIDE_IN_CSV}")
        df = pd.read_csv(CAR_WIDE_IN_CSV, parse_dates=["fomc_date"])
    else:
        raise FileNotFoundError(
            f"CAR data file not found: {CAR_WIDE_IN_PARQUET} or {CAR_WIDE_IN_CSV}.\n"
            "먼저 hw2_p2_event_study.py 를 실행해서 CAR 파일을 생성하세요."
        )

    # 날짜형 보정
    if not np.issubdtype(df["fomc_date"].dtype, np.datetime64):
        df["fomc_date"] = pd.to_datetime(df["fomc_date"])

    return df


# ------------------------------------------------------------
# 2. 특정 horizon에 대한 회귀용 데이터셋 만들기
# ------------------------------------------------------------

def build_regression_df(car_wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    주어진 horizon h에 대해 회귀에 쓸 패널 데이터 생성.
    - y = CAR_h
    - 회귀식: y_it = β1 * NS_t + μ_i + λ_t + ε_it
    """
    car_col = f"CAR{horizon}"
    if car_col not in car_wide.columns:
        raise ValueError(f"{car_col} column not found in CAR data.")

    df = car_wide.copy()

    df = df[["fic", "fomc_date", "NS", car_col]].copy()
    df = df.rename(columns={car_col: "y"})

    # 결측치 제거
    df = df.dropna(subset=["y", "NS"])

    print(f"[INFO] Horizon h={horizon}: #obs after dropping NA = {len(df)}")
    return df


# ------------------------------------------------------------
# 3. 패널 회귀 추정 (국가+이벤트 날짜 fixed effects)
# ------------------------------------------------------------

def run_fixed_effects_regression(df: pd.DataFrame, horizon: int):
    """
    회귀식:
      y_it = β1 * NS_t + μ_i + λ_t + ε_it
    구현:
      y ~ NS + C(fic) + C(fomc_date)
    표준오차:
      fomc_date 기준 클러스터링 (같은 이벤트 날짜 내 상관 허용)
    """
    # formula: y ~ NS + C(fic) + C(fomc_date)
    formula = "y ~ NS + C(fic) + C(fomc_date)"

    # patsy를 사용해 디자인 매트릭스 생성
    y, X = dmatrices(formula, data=df, return_type="dataframe")

    # statsmodels OLS
    model = sm.OLS(y, X)
    # 이벤트 날짜 기준으로 클러스터링
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["fomc_date"]})

    # 요약 출력
    print("=" * 80)
    print(f"Horizon h = {horizon}")
    print(result.summary().as_text())
    print("=" * 80)

    # summary를 파일로 저장
    summary_path = os.path.join(OUT_DIR, f"regression_summary_h{horizon}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())
    print(f"[INFO] Saved regression summary for h={horizon} to {summary_path}")

    # β1(NS)의 계수/표준오차/통계량만 뽑기
    if "NS" in result.params.index:
        beta1 = float(result.params["NS"])
        se1   = float(result.bse["NS"])
        t1    = float(result.tvalues["NS"])
        p1    = float(result.pvalues["NS"])
    else:
        # 이론상 발생하면 안 되지만, 안전장치
        beta1 = se1 = t1 = p1 = np.nan

    return {
        "horizon": horizon,
        "beta1_NS": beta1,
        "se_NS": se1,
        "t_NS": t1,
        "p_NS": p1,
        "nobs": int(result.nobs),
        "r2": float(result.rsquared)
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    # 1) CAR 데이터 로드
    car_wide = load_car_wide()

    # 2) 각 horizon(h=0~4)에 대해 회귀 추정
    results = []
    for h in range(0, 5):
        df_h = build_regression_df(car_wide, horizon=h)
        if len(df_h) == 0:
            print(f"[WARN] No observations for horizon h={h}. Skipping.")
            continue

        res = run_fixed_effects_regression(df_h, horizon=h)
        results.append(res)

    # 3) 결과 요약 테이블 저장
    if len(results) > 0:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("horizon").reset_index(drop=True)
        res_df.to_csv(OUT_RESULT_CSV, index=False)
        print(f"[INFO] Saved regression result summary to {OUT_RESULT_CSV}")
        print(res_df)
    else:
        print("[WARN] No regression results were produced (no horizons with data).")

    print("[DONE] Problem 3 regression analysis complete.")


if __name__ == "__main__":
    main()
