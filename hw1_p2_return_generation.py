# hw1_p2_return_generation.py
# Problem 2: Return Generating Process (최종 통합본)
# - 입력: data/hw1_firm_monthly_base.(csv|parquet)  ← 문제 1 결과(월말 기준)
# - 출력:
#   1) data/hw1_firm_monthly_enhanced.(csv|parquet)  # BOM/EOM 가중치 포함 보강본
#   2) data/hw1_country_monthly_returns.(csv|parquet)  # 국가별 EW, VW-BOM, VW-EOM
#   3) data/countries_returns/<ISO3>_country_monthly_returns.(csv|parquet)
#
# 규칙:
# - mret 결측/비정상 제거
# - VW_BOM: 전월말 시가총액 가중(look-ahead 방지, 기본)
# - VW_EOM: 당월말 시가총액 가중(참고)
# - 가중치는 각 (iso3, eom) 그룹에서 정규화; <=0 또는 NaN 가중은 제외
# - 터미널 프리뷰: 국가별 일부 행, 최근 3개월 요약, 가중치 커버리지, 샘플 TopN
# - (a)(b)(c) 리포트: mret 통계, mktcap/가중치 커버리지, 국가별 결과 요약

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
BASE_CANDIDATES = [
    DATA_DIR / "hw1_firm_monthly_base.parquet",
    DATA_DIR / "hw1_firm_monthly_base.csv",
]

OUT_FIRM_ENH_PQ = DATA_DIR / "hw1_firm_monthly_enhanced.parquet"
OUT_FIRM_ENH_CSV = DATA_DIR / "hw1_firm_monthly_enhanced.csv"
OUT_CTRY_PQ = DATA_DIR / "hw1_country_monthly_returns.parquet"
OUT_CTRY_CSV = DATA_DIR / "hw1_country_monthly_returns.csv"


def _load_base() -> pd.DataFrame:
    for p in BASE_CANDIDATES:
        if p.exists():
            print(f"[load] {p.name}")
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, parse_dates=["eom"])
    raise FileNotFoundError("Problem 1 결과 파일(data/hw1_firm_monthly_base.*)을 찾을 수 없습니다.")


def _require_cols(df: pd.DataFrame):
    required = ["eom", "mret"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"[error] 필수 열 누락: {c}")
    # 국가 식별 열 보정
    if "iso3" not in df.columns and "country" not in df.columns:
        raise RuntimeError("[error] 국가 식별 열(iso3 or country)이 필요합니다.")
    if "iso3" not in df.columns:
        df["iso3"] = df["country"]
    if "country" not in df.columns:
        df["country"] = df["iso3"]
    # 식별자 키 확인
    has_iid = "iid" in df.columns
    id_cols = ["gvkey", "iid"] if has_iid else ["gvkey"]
    for c in id_cols:
        if c not in df.columns:
            raise RuntimeError(f"[error] 식별자 열 누락: {c}")
    # 타입 정리
    df["eom"] = pd.to_datetime(df["eom"])
    df["iso3"] = df["iso3"].astype(str)
    df["country"] = df["country"].astype(str)
    return df, id_cols


def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _build_bom_weights(df: pd.DataFrame, id_cols):
    # 1) 전월말 시총 생성
    df = df.sort_values(id_cols + ["eom"])
    df["mktcap_bom"] = df.groupby(id_cols)["mktcap"].shift(1)

    # 2) 숫자형 강제 변환
    _coerce_numeric(df, ["mktcap", "mktcap_bom"])

    # 3) 유효 값(>0 & finite)만 가중치 후보
    bom_pos = df["mktcap_bom"].where(np.isfinite(df["mktcap_bom"]) & (df["mktcap_bom"] > 0))
    eom_pos = df["mktcap"].where(np.isfinite(df["mktcap"]) & (df["mktcap"] > 0))

    # 4) (iso3, eom) 합을 transform으로 계산 → 인덱스 불일치/Deprecation 해결
    denom_bom = bom_pos.groupby([df["iso3"], df["eom"]]).transform("sum")
    denom_eom = eom_pos.groupby([df["iso3"], df["eom"]]).transform("sum")

    df["w_vw_bom"] = bom_pos / denom_bom
    df["w_vw_eom"] = eom_pos / denom_eom

    return df


def _aggregate_country_monthly(df: pd.DataFrame) -> pd.DataFrame:
    # 유효 관측치 필터
    df = df.copy()
    _coerce_numeric(df, ["mret", "mktcap", "mktcap_bom", "w_vw_bom", "w_vw_eom"])
    df = df[pd.to_numeric(df["mret"], errors="coerce").notna()].copy()

    # EW: 단순 평균
    ew = (
        df.groupby(["iso3", "country", "eom"])
        .agg(EW=("mret", "mean"), N_firms=("mret", "size"))
        .reset_index()
    )

    # VW_BOM: 전월말 가중
    tmp_bom = df.dropna(subset=["w_vw_bom"])
    vw_bom = (
        tmp_bom.assign(ret_w=lambda x: x["mret"] * x["w_vw_bom"])
        .groupby(["iso3", "country", "eom"])
        .agg(VW_BOM=("ret_w", "sum"), BOM_coverage=("w_vw_bom", "count"))
        .reset_index()
    )

    # VW_EOM: 당월말 가중(참고)
    tmp_eom = df.dropna(subset=["w_vw_eom"])
    vw_eom = (
        tmp_eom.assign(ret_w=lambda x: x["mret"] * x["w_vw_eom"])
        .groupby(["iso3", "country", "eom"])
        .agg(VW_EOM=("ret_w", "sum"), EOM_coverage=("w_vw_eom", "count"))
        .reset_index()
    )

    out = (
        ew.merge(vw_bom, on=["iso3", "country", "eom"], how="left")
        .merge(vw_eom, on=["iso3", "country", "eom"], how="left")
        .sort_values(["iso3", "eom"])
        .reset_index(drop=True)
    )
    return out


def _save_dual(df: pd.DataFrame, pq_path: Path, csv_path: Path):
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(pq_path, index=False)
        print(f"[save] {pq_path}")
    except Exception as e:
        print(f"[warn] Parquet 저장 실패({e}) → CSV만 저장")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[save] {csv_path}")


def _split_by_country(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for iso3, sub in df.groupby("iso3", dropna=True):
        base = f"{iso3}_country_monthly_returns"
        pq_p = out_dir / f"{base}.parquet"
        csv_p = out_dir / f"{base}.csv"
        _save_dual(sub.sort_values(["eom"]), pq_p, csv_p)


def _print_console_preview(firm: pd.DataFrame, country: pd.DataFrame, rows: int = 8):
    # 보기 좋게 표시 옵션
    with pd.option_context("display.max_rows", rows,
                           "display.max_columns", 12,
                           "display.width", 120,
                           "display.float_format", "{:,.6f}".format):
        print("\n=== [미리보기] 국가별 월수익률 상위 몇 줄 ===")
        try:
            print(country.sort_values(["iso3", "eom"]).head(rows).to_string(index=False))
        except Exception:
            print("[알림] 미리보기를 표시할 수 없습니다.")

        # 최근 3개월치 국가별 요약
        eoms = country["eom"].dropna().sort_values().unique()
        recent_months = eoms[-3:] if len(eoms) >= 3 else eoms
        sub = country[country["eom"].isin(recent_months)].copy()
        if len(sub):
            print("\n=== [요약] 최근 3개월 국가별 EW / VW_BOM / VW_EOM ===")
            piv = (sub.pivot_table(index=["iso3","country","eom"],
                                   values=["EW","VW_BOM","VW_EOM"], aggfunc="mean")
                     .sort_values(["iso3","eom"]))
            print(piv.reset_index().to_string(index=False))
        else:
            print("\n[알림] 최근 3개월 뽑을 데이터가 충분하지 않습니다.")

        # 가중치 품질 간단 점검
        def _cov(s):
            s = pd.to_numeric(s, errors="coerce")
            return float(np.mean(np.isfinite(s)))
        bom_cov = _cov(firm["w_vw_bom"]) if "w_vw_bom" in firm.columns else 0.0
        eom_cov = _cov(firm["w_vw_eom"]) if "w_vw_eom" in firm.columns else 0.0
        print(f"\n=== [커버리지] 가중치 유효 비율 ===")
        print(f"BOM weight coverage: {bom_cov:.1%}  |  EOM weight coverage: {eom_cov:.1%}")

        # 한 국가·한 달 샘플로 가중치 상위 종목 Top N
        try:
            sample_idx = firm.dropna(subset=["w_vw_bom"]).sort_values(["iso3","eom"]).index
            if len(sample_idx):
                sample_row = firm.loc[sample_idx[0], ["iso3","eom"]]
                iso3_s, eom_s = sample_row["iso3"], pd.to_datetime(sample_row["eom"])
                topn = (firm[(firm["iso3"]==iso3_s) & (firm["eom"]==eom_s)]
                        .copy().sort_values("w_vw_bom", ascending=False).head(10))
                cols_show = [c for c in ["gvkey","iid","tic","conm","mret","mktcap_bom","w_vw_bom"] if c in topn.columns]
                print(f"\n=== [샘플] {iso3_s} {eom_s.date()} VW_BOM 상위 10 종목 ===")
                print(topn[cols_show].to_string(index=False))
            else:
                print("\n[알림] 샘플 가중치 프리뷰를 만들 수 없습니다(가중치 데이터 부족).")
        except Exception:
            print("\n[알림] 샘플 가중치 프리뷰 생성 중 오류가 발생했습니다.")


# ===== (a)(b)(c) 리포트 출력 =====
def _fmt_pct(x):
    try:
        return f"{float(x)*100:,.1f}%"
    except Exception:
        return "n/a"

def _print_abc_report(firm: pd.DataFrame, country: pd.DataFrame, last_k_months: int = 3, topn_weight: int = 10):
    print("\n==================== [Problem 2: (a)(b)(c) 결과 리포트] ====================")

    # (a) mret 점검
    print("\n(a) 월별 기업 수익률(mret) 점검")
    m = pd.to_numeric(firm["mret"], errors="coerce")
    valid = np.isfinite(m)
    print(f"- 관측치: {len(m):,}  |  유효 비율: {_fmt_pct(valid.mean())}")
    if valid.any():
        m_v = m[valid]
        print(f"- 기초 통계: mean={m_v.mean():.6f}, std={m_v.std():.6f}, min={m_v.min():.6f}, max={m_v.max():.6f}")
        print(f"- |mret| > 50%: {_fmt_pct((m_v.abs()>0.50).mean())}  |  >100%: {_fmt_pct((m_v.abs()>1.00).mean())}")

    # (b) 시가총액/가중치 커버리지
    print("\n(b) 시가총액 및 가중치(BOM/EOM) 커버리지")
    for col, label in [("mktcap", "EOM mktcap"), ("mktcap_bom", "BOM mktcap"),
                       ("w_vw_bom", "BOM weight"), ("w_vw_eom", "EOM weight")]:
        if col in firm.columns:
            s = pd.to_numeric(firm[col], errors="coerce")
            ok = np.isfinite(s) & (s > 0) if "mktcap" in col else np.isfinite(s)
            cov = ok.mean() if len(s) else np.nan
            print(f"- {label} 유효 비율: {_fmt_pct(cov)} (N={ok.sum():,}/{len(s):,})")
    if "w_vw_bom" in firm.columns:
        cnt = (firm["w_vw_bom"].notna().groupby([firm["iso3"], firm["eom"]]).sum())
        if len(cnt):
            print(f"- (참고) 국가-월 최소/중앙/최대 가중 종목수: "
                  f"{int(cnt.min())}/{int(cnt.median())}/{int(cnt.max())}")

    # (c) 국가별 월수익률 결과
    print("\n(c) 국가별 월수익률(EW, VW_BOM, VW_EOM)")
    if not country.empty:
        eoms = country["eom"].dropna().sort_values().unique()
        recent = eoms[-last_k_months:] if len(eoms) >= last_k_months else eoms
        sub = country[country["eom"].isin(recent)].copy()
        if len(sub):
            view_cols = [c for c in ["iso3","country","eom","EW","VW_BOM","VW_EOM","N_firms"] if c in sub.columns]
            print(f"- 최근 {len(recent)}개월 프리뷰:")
            with pd.option_context("display.max_rows", 200,
                                   "display.max_columns", 12,
                                   "display.width", 120,
                                   "display.float_format", "{:,.6f}".format):
                print(sub.sort_values(["iso3","eom"])[view_cols].to_string(index=False))
        g = (country.groupby(["iso3","country"])
                    .agg(EW_mean=("EW","mean"),
                         VW_BOM_mean=("VW_BOM","mean"),
                         VW_EOM_mean=("VW_EOM","mean"))
                    .reset_index()
                    .sort_values(["iso3"]))
        print("\n- 기간 평균(국가별):")
        with pd.option_context("display.max_rows", 200, "display.float_format", "{:,.6f}".format):
            print(g.to_string(index=False))
    else:
        print("- 국가별 결과가 비어 있습니다. (입력 데이터 확인 필요)")

    # 샘플 가중치 TOP N
    try:
        sample_idx = firm.dropna(subset=["w_vw_bom"]).sort_values(["iso3","eom"]).index
        if len(sample_idx):
            r = firm.loc[sample_idx[0], ["iso3","eom"]]
            iso3_s, eom_s = r["iso3"], pd.to_datetime(r["eom"])
            topn = (firm[(firm["iso3"]==iso3_s) & (firm["eom"]==eom_s)]
                    .copy().sort_values("w_vw_bom", ascending=False).head(topn_weight))
            cols_show = [c for c in ["gvkey","iid","tic","conm","mret","mktcap_bom","w_vw_bom"] if c in topn.columns]
            print(f"\n- 샘플({iso3_s} {eom_s.date()}) VW_BOM 상위 {topn_weight} 종목:")
            with pd.option_context("display.float_format", "{:,.6f}".format):
                print(topn[cols_show].to_string(index=False))
    except Exception:
        pass

    print("==========================================================================\n")


def main():
    print("1) 입력 로드…")
    firm = _load_base()
    if "eom" not in firm.columns:
        raise RuntimeError("eom(월말) 열이 필요합니다. (문제1 결과 확인)")

    print("2) 컬럼 점검/정리…")
    firm, id_cols = _require_cols(firm)
    if "mktcap" not in firm.columns:
        print("[warn] mktcap 열이 없어 계산 커버리지가 낮을 수 있습니다.")
        firm["mktcap"] = np.nan
    _coerce_numeric(firm, ["mret", "mktcap"])
    firm = firm[np.isfinite(firm["mret"])].copy()  # mret 실수형만 사용

    print("3) BOM/EOM 가중치 생성… (VW_BOM 기본)")
    firm = _build_bom_weights(firm, id_cols=id_cols)

    # 보강본 저장
    _save_dual(firm.sort_values(["iso3", "eom"]), OUT_FIRM_ENH_PQ, OUT_FIRM_ENH_CSV)

    print("4) 국가별 월수익률(EW, VW_BOM, VW_EOM) 집계…")
    country = _aggregate_country_monthly(firm)

    print("5) 결과 저장…")
    _save_dual(country, OUT_CTRY_PQ, OUT_CTRY_CSV)
    _split_by_country(country, DATA_DIR / "countries_returns")

    # 리포트
    def _cov(s):
        s = pd.to_numeric(s, errors="coerce")
        return float(np.mean(np.isfinite(s)))
    bom_cov = _cov(firm["w_vw_bom"])
    eom_cov = _cov(firm["w_vw_eom"])
    print(f"[report] rows(firm)={len(firm):,} | BOM weight coverage={bom_cov:.1%} | EOM weight coverage={eom_cov:.1%}")
    print(f"[report] rows(country-month)={len(country):,} | 기간: {country['eom'].min().date()} ~ {country['eom'].max().date()}")

    # ▼ 터미널 프리뷰 & (a)(b)(c) 리포트
    _print_console_preview(firm, country, rows=8)
    _print_abc_report(firm, country, last_k_months=3, topn_weight=10)


if __name__ == "__main__":
    main()
