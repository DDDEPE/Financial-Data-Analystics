# hw1_pipeline/pipeline.py
import pandas as pd

from .config import TARGET_ISO3, DATA_DIR
from .progress import step
from .transforms import asof_merge_monthly, build_marketcap_hints, add_labels
from .loaders import (
    load_price_base,
    load_dividends_monthly,     # (NEW) 배당 + mret 재구성
    attach_shares_and_mktcap,
    load_company_attrs,         # (수정) price_df 인자 받음
    load_funda_asof,
)

def load_firm_monthly(db):
    # 1) 가격/수익률 베이스
    with step("1) Load price/ret base"):
        price_df, price_meta = load_price_base(db)

    # 1.5) 배당 붙이고 mret 재구성
    with step("1.5) Attach dividends & reconstruct mret"):
        price2, div_meta = load_dividends_monthly(db, price_df)
        # mret 확정: 테이블 mret 우선, 결측만 재구성값으로 보충
        if "mret" in price2.columns:
            mret0 = pd.to_numeric(price2["mret"], errors="coerce")
            mret1 = pd.to_numeric(price2["mret_recon"], errors="coerce")
            price2["mret"] = mret0.where(mret0.notna(), mret1)
        else:
            price2["mret"] = pd.to_numeric(price2["mret_recon"], errors="coerce")
        price_df = price2

    # 2) 주식수 as-of + 기본 mktcap
    with step("2) Attach shares (as-of) & base mktcap"):
        firm, share_meta = attach_shares_and_mktcap(db, price_df, price_meta)

    # 3) 회사 속성 병합 (price_df 기반 부분조회/생략)
    with step("3) Merge company attributes"):
        attrs = load_company_attrs(db, price_df)
        on = ["gvkey"] + (["iid"] if ("iid" in firm.columns and "iid" in attrs.columns) else [])
        firm = firm.merge(attrs, on=on, how="left")

    # 4) 펀더멘털 as-of 병합
    with step("4) Load fundamentals & as-of merge"):
        funda = load_funda_asof(db)
        firm  = asof_merge_monthly(firm, funda)

    # 5) 라벨
    with step("5) Add labels"):
        firm = add_labels(firm)

    # 6) 마켓캡 힌트
    with step("6) Build market cap hints"):
        firm = build_marketcap_hints(firm)

    # 표준 컬럼 정리
    keep = ["country","iso3","msci_group","gvkey","iid","eom","year","month",
            "mret","px","shr","mktcap","divps_m",
            "mktcap_from_monthly","mktcap_from_mkvalt","mktcap_from_qtr",
            "mktcap_pref","mktcap_source",
            "ratio_pref_vs_monthly","ratio_pref_vs_mkvalt","ratio_pref_vs_qtr",
            "conm","tic","isin","sedol","sic","gsector","ggroup","gind","subind",
            "curcd","exchg","loc","fic","cout",
            "datadate","sale","sales","revt","at","ceq","lt","ebit","oibdp","ni","capx",
            "prccq","prcc_f","prcc","cshoq","csho","mkvaltq","mkvalt"]
    keep = [c for c in keep if c in firm.columns]
    out = (firm[keep]
           .sort_values(["country","gvkey"] + (["iid"] if "iid" in firm.columns else []) + ["eom"])
           .reset_index(drop=True))

    # 진단
    mcap_pref = pd.to_numeric(out.get("mktcap_pref", out.get("mktcap")), errors="coerce")
    valid = mcap_pref.notna(); denom = int(valid.sum())
    mcap_ratio = ((mcap_pref > 0) & valid).sum() / denom if denom > 0 else 0.0
    print(f"[firm_monthly+] rows={len(out):,} | "
          f"firms={out[['gvkey'] + (['iid'] if 'iid' in out.columns else [])].drop_duplicates().shape[0]}")
    print(f"[firm_monthly+] valid preferred mktcap ratio: {float(mcap_ratio):.2%}")

    if (pd.to_numeric(out["mret"], errors="coerce").notna().sum() == 0):
        print("[주의] mret가 전부 결측입니다. 배당/가격/전월가격 확인 필요.")

    return out
