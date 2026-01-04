# hw2_p1_data_construction.py
# ------------------------------------------------------------
# HW2 Problem 1: Data Construction (1995-2024)
#
# - dwcountryreturns.csv: 국가별 일별 수익률 사용
# - FTSE All-World Index.csv: 글로벌 월드 인덱스 가격 → 일별 수익률 생성
# - ABJ-2024-monetary-policy-surprises.xlsx: NS shock 병합
# - 샘플: NS shock 커버 기간(1995-02-01 ~ 2024-05-01) 전체를 덮는 국가만 사용
# - 타임존 반영:
#     * 미주권 국가(BRA, CHL, COL, MEX): 발표 당일이 event day 0
#     * 그 외 국가는 발표 다음 거래일이 event day 0
#     * 비거래일이면 다음 거래일로 이동
# - 출력: out/event_panel_day0_to_p4.parquet (또는 CSV fallback)
# ------------------------------------------------------------

import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd

# -------------------------
# 설정
# -------------------------
START_DATE = "1995-01-01"  # 편의상 리밋용
END_DATE   = "2024-12-31"

IN_FILE_RETURNS = "dwcountryreturns.csv"
IN_FILE_SHOCKS  = "ABJ-2024-monetary-policy-surprises.xlsx"
IN_FILE_WORLD   = "FTSE All-World Index.csv"

OUT_DIR = "out"
OUT_PARQUET = os.path.join(OUT_DIR, "event_panel_day0_to_p4.parquet")
OUT_CSV     = os.path.join(OUT_DIR, "event_panel_day0_to_p4.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# 미주권 FIC (발표 당일을 event day 0으로 사용)
AMERICAS_FIC = {
    "BRA", "CHL", "COL", "MEX",  # 이 데이터셋에 실제로 존재하는 미주국가
    "USA", "CAN", "ARG", "PER"   # 혹시 나중에 추가될 수 있는 코드들
}

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
# 1. 국가별 일별 수익률 로드
# ------------------------------------------------------------

def load_country_returns(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading country returns from {path} ...")
    df = pd.read_csv(path)

    # date → datetime
    if "date" not in df.columns:
        raise ValueError("dwcountryreturns.csv 에 'date' 컬럼이 없습니다.")
    df["date_dt"] = pd.to_datetime(df["date"].astype(str))

    # 기본 수익률 컬럼 선택: portret (local) 또는 portretx (USD) 중 하나
    ret_col = None
    if "portret" in df.columns:
        ret_col = "portret"
    elif "portretx" in df.columns:
        ret_col = "portretx"
    else:
        raise ValueError("dwcountryreturns.csv 에서 'portret' 또는 'portretx' 수익률 컬럼을 찾을 수 없습니다.")

    df = df.rename(columns={ret_col: "ret"})
    print(f"[INFO] Using return column: {ret_col} → 'ret'")

    # 샘플 기간으로 1차 필터 (여유있게)
    df = df[(df["date_dt"] >= START_DATE) & (df["date_dt"] <= END_DATE)].copy()

    return df


# ------------------------------------------------------------
# 2. FTSE All-World Index에서 글로벌 월드 수익률 생성
# ------------------------------------------------------------

def load_world_index(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading world index from {path} ...")
    w = pd.read_csv(path)

    # 날짜 컬럼 찾기 (보통 datadate)
    date_col = _find_column(w, ["datadate", "date"])
    price_col = _find_column(w, ["prccd", "price", "index_price"])

    w["date_dt"] = pd.to_datetime(w[date_col])
    w = w.sort_values("date_dt").copy()

    # 일별 단순 수익률: r_t = (P_t / P_{t-1}) - 1
    w["world_ret"] = w[price_col].pct_change()

    # 샘플 기간 필터
    w = w[(w["date_dt"] >= START_DATE) & (w["date_dt"] <= END_DATE)].copy()

    world_ret = w[["date_dt", "world_ret"]].dropna().copy()
    print(f"[INFO] World index sample: {world_ret['date_dt'].min().date()} ~ {world_ret['date_dt'].max().date()}")
    return world_ret


# ------------------------------------------------------------
# 3. NS Shock 로드
# ------------------------------------------------------------

def load_ns_shocks(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading NS shocks from {path} ...")
    xls = pd.ExcelFile(path)
    # 보통 'Data' 시트에 있음
    sheet_name = "Data"
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"{path} 에 'Data' 시트를 찾을 수 없습니다. 시트 이름을 확인하세요.")

    df = pd.read_excel(xls, sheet_name=sheet_name)

    # 날짜/NS/GSS 컬럼 찾기
    date_col = _find_column(df, ["date", "Date"])
    ns_col   = _find_column(df, ["NS", "ns"])
    gss_tgt  = _find_column(df, ["GSS_target", "GSS Target"])
    gss_path = _find_column(df, ["GSS_path", "GSS Path"])

    df["fomc_date"] = pd.to_datetime(df[date_col])
    df = df.rename(columns={
        ns_col: "NS",
        gss_tgt: "GSS_target",
        gss_path: "GSS_path"
    })

    df = df[["fomc_date", "NS", "GSS_target", "GSS_path"]].copy()
    df = df.sort_values("fomc_date")

    print(f"[INFO] NS shock sample: {df['fomc_date'].min().date()} ~ {df['fomc_date'].max().date()}")
    return df


# ------------------------------------------------------------
# 4. NS shock 기간 전체를 커버하는 국가만 선택
# ------------------------------------------------------------

def select_fully_covered_countries(returns: pd.DataFrame, shocks: pd.DataFrame) -> List[str]:
    shock_start = shocks["fomc_date"].min()
    shock_end   = shocks["fomc_date"].max()

    coverage = returns.groupby("fic")["date_dt"].agg(["min", "max"])
    eligible = coverage[(coverage["min"] <= shock_start) & (coverage["max"] >= shock_end)].copy()

    print("[INFO] Countries with full coverage over NS shock sample:")
    for fic, row in eligible.iterrows():
        print(f"  {fic}: {row['min'].date()} ~ {row['max'].date()}")

    return sorted(eligible.index.tolist())


# ------------------------------------------------------------
# 5. 이벤트 패널 생성 (event day 0~4)
# ------------------------------------------------------------

def build_event_panel(
    returns: pd.DataFrame,
    shocks: pd.DataFrame,
    world: pd.DataFrame,
    americas_fic: set,
    max_horizon: int = 4
) -> pd.DataFrame:
    # 월드 수익률 병합
    returns = returns.merge(world, on="date_dt", how="left")

    # 샘플 기간을 NS shock 기간으로 제한
    shock_start = shocks["fomc_date"].min()
    shock_end   = shocks["fomc_date"].max()
    returns = returns[(returns["date_dt"] >= shock_start) & (returns["date_dt"] <= shock_end + pd.Timedelta(days=10))].copy()

    rows = []

    # 국가별로 trading calendar 기반으로 이벤트 윈도우 구성
    for fic, g in returns.groupby("fic"):
        g = g.sort_values("date_dt").reset_index(drop=True)
        dates = g["date_dt"].to_numpy()
        is_americas = fic in americas_fic

        print(f"[INFO] Building event windows for {fic} (Americas={is_americas}), #trading days={len(dates)}")

        for _, ev in shocks.iterrows():
            fomc_date = ev["fomc_date"]

            # 타임존 룰:
            # - 미주권: 발표 당일 (fomc_date) 이후 첫 거래일 (대부분 fomc_date 자체가 거래일)
            # - 그 외: 발표 '다음 날' 이후 첫 거래일
            if is_americas:
                target = fomc_date
            else:
                target = fomc_date + pd.Timedelta(days=1)

            target64 = np.datetime64(target.normalize())
            pos0 = dates.searchsorted(target64)  # 첫 date_dt >= target

            if pos0 >= len(dates):
                continue  # 이후 더 이상 거래일이 없으면 skip

            # 이벤트 윈도우가 끝까지 0~max_horizon 존재하는지 확인
            if pos0 + max_horizon >= len(dates):
                continue

            for offset in range(max_horizon + 1):
                pos = pos0 + offset
                row = g.iloc[pos]

                rows.append({
                    "fic": row["fic"],
                    "country": row["country"],
                    "ret_date": row["date_dt"],
                    "event_day": offset,          # 0,1,2,3,4
                    "fomc_date": fomc_date,
                    "NS": ev["NS"],
                    "GSS_target": ev["GSS_target"],
                    "GSS_path": ev["GSS_path"],
                    "ret": row["ret"],
                    "world_ret": row["world_ret"],
                    "currency": row.get("currency", None)
                })

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["fic", "fomc_date", "event_day"]).reset_index(drop=True)

    print(f"[INFO] Event panel constructed: {panel.shape[0]} rows")
    return panel


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    # 1) 로드
    returns = load_country_returns(IN_FILE_RETURNS)
    shocks  = load_ns_shocks(IN_FILE_SHOCKS)
    world   = load_world_index(IN_FILE_WORLD)

    # 2) NS shock 기간 전체 커버하는 국가만 선택
    eligible_fic = select_fully_covered_countries(returns, shocks)
    print(f"[INFO] Eligible countries (full coverage): {eligible_fic}")

    returns = returns[returns["fic"].isin(eligible_fic)].copy()

    # 3) 이벤트 패널 생성
    panel = build_event_panel(
        returns=returns,
        shocks=shocks,
        world=world,
        americas_fic=AMERICAS_FIC,
        max_horizon=4
    )

    # 4) 저장 (Parquet 우선, 안 되면 CSV fallback)
    try:
        panel.to_parquet(OUT_PARQUET, index=False)
        print(f"[INFO] Saved event panel to {OUT_PARQUET}")
    except Exception as e:
        print(f"[WARN] Failed to save Parquet ({e}), saving CSV instead.")
        panel.to_csv(OUT_CSV, index=False)
        print(f"[INFO] Saved event panel to {OUT_CSV}")

    print("[DONE] Problem 1 data construction complete.")


if __name__ == "__main__":
    main()
