# 공정제어 상시모니터링 시스템

철강/금속 압연·코일 생산라인 공정 이상탐지 및 SPC 관리 대시보드

## 주요 기능

| 메뉴 | 설명 |
|------|------|
| 라인별 현황 | A·B·C 라인 실시간 6개 공정변수 상태 카드 |
| 변수별 추이 | 기간 선택, 이상 포인트 마킹, 분포/박스플롯 |
| 이상탐지 알림 | Z-Score/IQR/Isolation Forest 결과, 심각도별 분류 |
| 설비별 위험평가 | 5개 설비 × 3개 라인 위험점수 히트맵 |
| SPC 관리도 | X-bar/R 관리도 + Cp/Cpk 공정능력지수 |
| 결함 트렌드 | 결함률 추이, 온도 상관, 시간대별 히트맵 |
| 데이터 수집 | 센서별 수집 현황·품질 모니터링 |
| 도움말 | 3단 포맷(활용방법·판단기준·실무적용) |

## 이상탐지 알고리즘

- **Z-Score**: 단변량 통계 이상치 (μ±3σ)
- **IQR**: 사분위 범위 이상치 (Q1-1.5·IQR ~ Q3+1.5·IQR)
- **Isolation Forest**: 다변량 복합 이상 (6개 변수 동시)
- **SPC X-bar/R 관리도**: 서브그룹 평균·범위 통계 관리
- **Cp/Cpk**: 공정능력지수 (규격 대비 공정 여유)

## 공정 변수

| 변수 | 정상 범위(A라인) | 경보 기준 |
|------|-----------------|-----------|
| 압연 온도(℃) | 850 ~ 950 | UCL 950 / LCL 850 |
| 압연 압력(ton) | 245 ~ 315 | UCL 315 / LCL 245 |
| 라인 속도(m/min) | 36 ~ 54 | UCL 54 / LCL 36 |
| 두께 편차(mm) | ±0.08 이내 | 공차 초과 |
| 결함률(%) | 2.0% 이하 | 2.0% 초과 |
| 진동(Hz) | 20.0 이하 | 20.0 초과 |

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포

1. GitHub 신규 레포 생성: `process_monitoring`
2. 이 폴더 전체 push
3. [share.streamlit.io](https://share.streamlit.io) → New app → 레포 연결
4. Main file: `app.py`

## 데이터 구조

현재 버전은 30일분 샘플 데이터를 인메모리로 생성합니다.
실제 PLC/센서 연동은 `data_generator.py`의 `generate_process_data()`를
실시간 데이터 소스로 교체하여 확장합니다.

## 기술 스택

- **대시보드**: Streamlit + Plotly
- **이상탐지**: scikit-learn (IsolationForest), SciPy (Z-score)
- **데이터**: pandas, numpy (샘플 데이터 생성)
