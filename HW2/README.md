# 📈 Global Stock Market Reactions to U.S. Monetary Policy (NS Shock)

본 저장소는 **미국 연준(Fed)의 통화정책 발표가 글로벌 주식시장에 미치는 영향**을  
고빈도 통화정책 충격 지표인 **Nakamura & Steinsson (2018)의 NS Shock**을 활용하여  
이벤트 스터디(Event Study)와 패널 회귀 분석으로 실증 분석한 프로젝트이다.

본 프로젝트는 성균관대학교 핀테크융합학과  
**「금융데이터분석(Financial Data Analysis)」 수업 과제(Homework 2)**의 일환으로 수행되었다.

---

## 🔍 Research Question

- 미국 통화정책(NS Shock)은 글로벌 주식시장에 **어떤 방향의 반응**을 유발하는가?
- 해당 반응은 **즉각적인가, 아니면 며칠간 지속되는가?**
- 금리효과가 아닌 **정보효과(Information Effect)**가 지배적인가?

---

## 🧠 Methodology Overview

본 분석은 다음과 같은 **3단계 분석 파이프라인**으로 구성된다.

1. **Data Construction**
   - 국가별 일별 주식수익률, 세계시장 수익률, NS shock을 결합한 이벤트 패널 구성
   - 국가별 시간대(Time Zone)를 고려하여 event day 0 정의

2. **Event Study**
   - 글로벌 시장모형(Global Market Model)을 이용해 기대수익률 추정
   - 비정상수익률(AR) 및 누적비정상수익률(CAR) 계산

3. **Panel Regression**
   - 국가 고정효과 + 이벤트 날짜 고정효과 패널 회귀
   - NS shock의 인과적 효과 추정 (clustered standard errors)

---

## 📂 Repository Structure

.
├── hw2_p1_data_construction.py # Problem 1: 데이터 구성 (이벤트 패널 생성)
├── hw2_p2_event_study.py # Problem 2: AR / CAR 계산
├── hw2_p3_regression.py # Problem 3: 패널 회귀 분석
├── out/ # 회귀 결과 및 중간 산출물
│ ├── event_panel_day0_to_p4.parquet
│ ├── event_panel_with_ar_car.parquet
│ ├── event_car_wide.parquet
│ ├── regression_results_ns_by_horizon.csv
│ └── regression_summary_h*.txt
└── README.md

---

## 🧩 Core Files Explanation (핵심 4개 기준)

### 1️⃣ `hw2_p1_data_construction.py`
**이벤트 스터디의 기반이 되는 패널 데이터 생성**

- 국가별 일별 수익률 + FTSE All-World Index + NS Shock 결합
- Full Coverage Rule 적용 (NS shock 전 기간을 커버하는 국가만 사용)
- 미주권 / 비미주권 시간대 차이를 반영한 event day 0 조정
- event day 0 ~ +4 거래일 윈도우 생성

**Output**
- `event_panel_day0_to_p4.parquet`

---

### 2️⃣ `hw2_p2_event_study.py`
**글로벌 시장모형 기반 AR / CAR 계산**

- 글로벌 시장모형 추정  
  \[
  R_{it} = \alpha_i + \beta_i R_{world,t} + \varepsilon_{it}
  \]
- 비정상수익률(AR) 및 누적비정상수익률(CAR0~CAR4) 산출
- long / wide 포맷 데이터 동시 생성

**Output**
- `event_panel_with_ar_car.parquet`
- `event_car_wide.parquet`

---

### 3️⃣ `hw2_p3_regression.py`
**NS Shock의 인과효과를 추정하는 패널 회귀 분석**

회귀식:
\[
CAR_{i,t}^{(h)} = \beta_1 NS_t + \mu_i + \lambda_t + \varepsilon_{i,t}
\]

- 국가 고정효과(Country FE)
- 이벤트 날짜 고정효과(Event-date FE)
- 표준오차는 FOMC 날짜 기준 클러스터링

**Output**
- `regression_results_ns_by_horizon.csv`
- `regression_summary_h0.txt` ~ `h4.txt`

---

### 4️⃣ README / Report
**분석 구조, 경제적 해석, 결과 요약을 설명하는 문서 레이어**

- 이벤트 스터디 설계 논리
- NS Shock의 정보효과(Information Effect)
- 글로벌 전이 메커니즘 해석

---

## 📊 Key Results (Summary)

| Horizon | β₁ (NS Shock) | Interpretation |
|------|--------------|---------------|
| CAR0 | +0.0098*** | 발표 직후 약 +1% |
| CAR1 | +0.0113*** | 다음날 반응 확대 |
| CAR2 | +0.0094*** | 효과 지속 |
| CAR3 | +0.0081** | 점진적 완화 |
| CAR4 | +0.0090** | 정보효과 유지 |

- 모든 horizon에서 **양(+)의 반응 & 통계적 유의**
- NS shock은 금리 인상/인하 그 자체보다  
  **연준의 경기 전망 정보 전달 효과**가 지배적임을 시사

---

## 📌 Conclusion

본 프로젝트는 금융데이터분석 수업에서 다룬 이론과 기법을  
**실제 국제금융 논문 수준의 분석 파이프라인**으로 구현하였다.

- 정교한 이벤트 패널 구성 (Time Zone Adjustment 포함)
- 글로벌 시장모형 기반 AR / CAR 계산
- 국가·이벤트 고정효과 패널 회귀를 통한 인과효과 추정

**결론적으로, 미국의 NS Shock은 글로벌 주식시장에  
즉각적이며 지속적인 양(+)의 영향을 미치며,  
이는 Nakamura & Steinsson (2018)의 Information Effect 가설과 정합적이다.**

---

## 📚 Reference

- Nakamura, E., & Steinsson, J. (2018).  
  *High-Frequency Identification of Monetary Non-Neutrality:  
  The Information Effect.*  
  **Quarterly Journal of Economics, 133(3), 1283–1330.**
