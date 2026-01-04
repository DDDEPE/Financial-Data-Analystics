# 금융데이터분석 Replication Project  
## Asset Growth and the Cross-Section of Stock Returns

본 저장소는 Cooper, Gulen, Schill (2008),  
*Asset Growth and the Cross-Section of Stock Returns* (Journal of Finance) 논문의  
주요 실증 결과를 재현(replication)하고,  
동일한 분석 방법론을 최근 데이터(2004–2024년)에 적용하여  
자산성장률(asset growth)과 미래 주식수익률 간 관계의 지속성을 검증하는 것을 목적으로 한다.

본 연구는 대학원 **금융데이터분석(Financial Data Analysis)** 과목의 조별 과제로 수행되었으며,  
실증 금융(empirical finance) 연구의 재현 가능성(reproducibility)과  
데이터 확장에 따른 결과의 안정성(robustness)을 확인하는 데 초점을 둔다.

---

## 1. 연구 배경 및 목적 (Motivation)

Cooper, Gulen, Schill (2008)은 기업의 자산성장률(asset growth)이  
향후 주식수익률과 유의한 음(-)의 관계를 가진다는 사실을 보이며,  
투자(investment) 관련 기업 의사결정이 자본시장에서 체계적으로 가격화됨을 제시하였다.

본 프로젝트는 다음의 두 가지 연구 질문에 답하고자 한다.

1. 원 논문에서 보고된 자산성장률 효과가  
   현재의 데이터 환경에서도 동일하게 관찰되는가?
2. 2004년 이후 최근 기간을 포함하였을 때,  
   해당 효과의 크기 및 통계적 유의성은 어떻게 변화하는가?

이를 위해 원 논문의 실증 절차를 최대한 충실히 재현한 후,  
동일한 변수 정의와 분석 구조를 유지한 채 표본 기간을 확장한다.

---

## 2. 데이터 및 표본 구성 (Data and Sample Construction)

본 연구는 WRDS를 통해 제공되는 Compustat 및 CRSP 데이터를 사용한다.

- **Compustat (Fundamentals Annual)**  
  기업의 재무제표 정보를 이용하여 자산성장률 및 회계 변수 산출
- **CRSP (Monthly Stock File)**  
  월별 주식수익률 및 시가총액 정보 활용

원 논문과 동일하게 CRSP–Compustat 매칭을 수행하였으며,  
분석의 재현성과 실행 효율성을 위해 주요 데이터는 CSV 형태로 저장하였다.

표본 기간은 다음과 같이 구성된다.

- **Replication Sample**: 1963–2003  
- **Extended Sample**: 1963–2024  

---

## 3. 분석 방법론 (Empirical Methodology)

실증 분석은 원 논문과 동일한 두 가지 접근법을 따른다.

1. **Summary Statistics (Table 1)**  
   주요 변수들의 분포적 특성을 요약하여 표본의 전반적인 특성을 파악
2. **Fama–MacBeth Cross-Sectional Regressions (Table 3)**  
   자산성장률이 미래 주식수익률에 미치는 영향을 통제변수와 함께 분석

Table 3에서는 전체 표본(All Firms)과 소형주(Small Firms)를 구분하여 분석하며,  
추가적인 패널 분석을 통해 결과의 강건성을 검증한다.

---

## 4. 폴더 구조 (Repository Structure)
금융데이터분석_Replication/
├─ Asset Growth and the Cross-Section of Stock Returns.pdf
├─ Final Report_장현민, 신현민, 이수민.pdf
├─ Table 1. Summary Statistics/
│ ├─ Table 1_1963_2003.py
│ ├─ Table 1_1963_2024.py
│ └─ Table 1_data/
│ ├─ Compustat_funda_1963_2003.csv
│ ├─ Compustat_funda_1963_2024.csv
│ ├─ Crsp_compustat_link_1963_2003.csv
│ ├─ Crsp_compustat_link_1963_2024.csv
│ ├─ Crsp_monthly_1963_2003.csv
│ └─ Crsp_monthly_1963_2024.csv
└─ Table 3. Main Tables/
├─ Panel A&B/
│ └─ data_project_table3_mark3.py
└─ Panel C&D/
├─ Panel C&D data.csv
├─ Table 3_Panel C&D_1963_2003.ipynb
└─ Table 3_Panel C&D_1963_2024.ipynb

---

## 5. 실증 분석 구현 (Empirical Implementation)

### 5.1 Summary Statistics의 구성 (Table 1)

Table 1은 자산성장률을 포함한 주요 변수들의  
기초 통계량을 산출하여 표본의 특성을 요약하는 것을 목적으로 한다.

- `Table 1_1963_2003.py`  
  → 원 논문 표본 기간(1963–2003)에 대한 요약통계 재현
- `Table 1_1963_2024.py`  
  → 동일한 변수 정의를 유지한 채 최근 데이터까지 확장

이를 통해 원 논문 결과와 확장 표본 간의 분포적 차이를 비교할 수 있다.

---

### 5.2 Fama–MacBeth 회귀 분석 (Table 3 Panel A & B)

Table 3 Panel A 및 Panel B에서는  
Fama–MacBeth (1973) 방법론을 이용한 횡단면 회귀 분석을 수행한다.

- Panel A: 전체 기업(All Firms)
- Panel B: 소형주(Small Firms)

자산성장률(asset growth)을 핵심 설명변수로 사용하며,  
원 논문에서 제시된 통제변수들을 함께 포함한다.

해당 분석은 `data_project_table3_mark3.py` 파일을 통해 수행된다.

---

### 5.3 추가 패널 분석 및 기간 확장 (Table 3 Panel C & D)

Panel C와 Panel D는 자산성장률 효과의 강건성을 검증하기 위한 추가 분석이다.

- `Table 3_Panel C&D_1963_2003.ipynb`  
  → 원 논문 기간 Panel C/D 결과 재현
- `Table 3_Panel C&D_1963_2024.ipynb`  
  → 확장 표본에 대한 동일 분석 수행

이를 통해 자산성장률과 미래 주식수익률 간 관계가  
최근 기간에도 유지되는지를 확인한다.

---

## 6. 재현 결과에 대한 유의사항 (Notes on Replication)

WRDS 데이터의 특성상 다음과 같은 이유로  
원 논문과 완전히 동일한 수치가 도출되지 않을 수 있다.

- 데이터 수정(restatement) 및 업데이트
- CRSP–Compustat 매칭 방식의 차이
- delisting return 처리 방식의 차이

그럼에도 불구하고,  
자산성장률과 미래 주식수익률 간의 음(-)의 관계라는  
핵심적인 경제적 해석은 원 논문과 일관되게 관찰된다.

---

## 7. 연구진 (Authors)

- 장현민  
- 신현민  
- 이수민  

---

## 8. 참고문헌 (Reference)

Cooper, M. J., Gulen, H., & Schill, M. J. (2008).  
**Asset Growth and the Cross-Section of Stock Returns.**  
*Journal of Finance*, 63(4), 1609–1651.
