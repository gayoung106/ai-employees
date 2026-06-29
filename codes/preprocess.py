from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat


# ============================================================
# 0. 경로 설정
# - 이 파일(preprocess.py)을 ai-employees 프로젝트 루트에 둡니다.
#
# ai-employees/
# ├─ preprocess.py
# ├─ datas/
# │  └─ raw_data.sav
# └─ clean_data/
#    └─ clean_data.parquet   # 실행 후 자동 생성
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_PATH = PROJECT_ROOT / "datas" / "raw_data.sav"
OUTPUT_DIR = PROJECT_ROOT / "clean_data"
OUTPUT_PATH = OUTPUT_DIR / "clean_data.parquet"


# ============================================================
# 1. 사용할 원본 변수
# ============================================================

RAW_COLUMNS = [
    "uniqid",       # 응답자 식별자
    "country",      # 국가/표본 ID
    "isocntry",     # 국가 ISO 코드
    "w1",           # 가중치

    "d15a",         # 현재 직업
    "sd28",         # 산업/부문
    "sd29",         # 조직 규모

    "d11",          # 연령
    "d10",          # 성별
    "d8",           # 전일제 교육 종료 연령
    "qb2_2",        # 직무 수행 디지털 역량

    "qb7_4",        # 자동화 감시 경험
    "qb7_5",        # 자동화 성과평가·제재·보상 경험

    "qb9.1",        # 단순 고지
    "qb9.2",        # 상세 설명
    "qb9.3",        # 개인정보 접근권
    "qb9.4",        # 자동화 분석 결과 접근권
    "qb9.5",        # 무고지
    "qb9.6",        # 해당 없음
    "qb9.7",        # 모름
]


# ============================================================
# 2. 보조 함수
# ============================================================

def ensure_columns_exist(data: pd.DataFrame, required_columns: list[str]) -> None:
    """필수 변수가 SAV 파일에 모두 존재하는지 확인합니다."""
    missing_columns = sorted(set(required_columns) - set(data.columns))

    if missing_columns:
        raise KeyError(
            "SAV 파일에 필요한 변수가 없습니다:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )


def to_valid_numeric(
    series: pd.Series,
    valid_values: list[int] | range | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
) -> pd.Series:
    """유효 범위 밖의 값을 결측치로 처리합니다."""
    cleaned = pd.to_numeric(series, errors="coerce")

    if valid_values is not None:
        cleaned = cleaned.where(cleaned.isin(valid_values))

    if min_value is not None:
        cleaned = cleaned.where(cleaned >= min_value)

    if max_value is not None:
        cleaned = cleaned.where(cleaned <= max_value)

    return cleaned


def make_exposure_variable(series: pd.Series) -> pd.Series:
    """
    QB7 문항 재코딩
    1 Yes, all the time -> 1
    2 Yes, often        -> 1
    3 No, rarely        -> 0
    4 No, never         -> 0
    5 Not applicable    -> 결측
    6 Don't know        -> 결측
    """
    mapping = {
        1.0: 1,
        2.0: 1,
        3.0: 0,
        4.0: 0,
    }

    return series.map(mapping).astype("Int64")


# ============================================================
# 3. 원자료 불러오기
# ============================================================

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"원자료 파일을 찾을 수 없습니다.\n"
        f"확인 경로: {RAW_PATH}"
    )

raw_df, metadata = pyreadstat.read_sav(
    RAW_PATH,
    usecols=RAW_COLUMNS,
)

ensure_columns_exist(raw_df, RAW_COLUMNS)

print("=" * 60)
print("원자료 로드 완료")
print(f"원자료 행 수: {len(raw_df):,}")
print("=" * 60)


# ============================================================
# 4. 분석 표본 제한
# ============================================================

# 현재 임금근로자만 유지
# d15a = 10~18
df = raw_df.loc[
    raw_df["d15a"].between(10, 18, inclusive="both")
].copy()

n_after_employed_filter = len(df)

# 공공부문(5) vs 비공공부문(1~4)만 유지
# 6=Other, 7=Don't know, 결측은 제외
df = df.loc[
    df["sd28"].isin([1, 2, 3, 4, 5])
].copy()

n_after_sector_filter = len(df)

# QB9에서 "Not applicable" 또는 "Don't know" 응답 제외
df = df.loc[
    (df["qb9.6"] != 1)
    & (df["qb9.7"] != 1)
].copy()

n_final = len(df)


# ============================================================
# 5. 분석용 변수 생성
# ============================================================

clean_df = pd.DataFrame(index=df.index)

# ------------------------------------------------------------
# 5-1. 식별자·국가·가중치
# ------------------------------------------------------------

clean_df["respondent_id"] = df["uniqid"]
clean_df["country_raw"] = df["isocntry"]

# 독일 동·서부 표본은 국가 고정효과 분석에서 하나의 독일로 통합
clean_df["country_fe"] = (
    df["isocntry"]
    .replace(
        {
            "DE-E": "DE",
            "DE-W": "DE",
        }
    )
)

clean_df["country_id"] = df["country"]
clean_df["weight_w1"] = pd.to_numeric(df["w1"], errors="coerce")


# ------------------------------------------------------------
# 5-2. 핵심 독립변수: 공공부문 여부
# ------------------------------------------------------------

clean_df["sector_code"] = df["sd28"].astype("Int64")

# sd28 = 5: Public sector
# sd28 = 1~4: non-public sector
clean_df["public_sector"] = (
    df["sd28"]
    .eq(5)
    .astype("int8")
)


# ------------------------------------------------------------
# 5-3. 핵심 종속변수: 정보공개·접근권
# ------------------------------------------------------------

# QB9.2~QB9.5는 각각 독립적인 0/1 변수입니다.
# 상세 설명, 개인정보 접근권, 자동화 분석 결과 접근권은 복수응답이 가능합니다.

clean_df["detailed_explanation"] = (
    pd.to_numeric(df["qb9.2"], errors="coerce")
    .astype("Int64")
)

clean_df["personal_data_access"] = (
    pd.to_numeric(df["qb9.3"], errors="coerce")
    .astype("Int64")
)

clean_df["automated_analysis_access"] = (
    pd.to_numeric(df["qb9.4"], errors="coerce")
    .astype("Int64")
)

clean_df["not_informed"] = (
    pd.to_numeric(df["qb9.5"], errors="coerce")
    .astype("Int64")
)

# 참고용 변수: 단순 고지
clean_df["basic_notification_only"] = (
    pd.to_numeric(df["qb9.1"], errors="coerce")
    .astype("Int64")
)


# ------------------------------------------------------------
# 5-4. 보조·강건성 분석 변수: 알고리즘 관리 경험
# ------------------------------------------------------------

clean_df["automated_monitoring_exposure"] = make_exposure_variable(
    df["qb7_4"]
)

clean_df["automated_performance_management_exposure"] = make_exposure_variable(
    df["qb7_5"]
)


# ------------------------------------------------------------
# 5-5. 통제변수
# ------------------------------------------------------------

# 연령: 15~98세만 유효값으로 유지
clean_df["age"] = to_valid_numeric(
    df["d11"],
    min_value=15,
    max_value=98,
)

# 성별
# 1=Man, 2=Woman, 3=None of the above / Non binary / Prefer not to say
clean_df["gender"] = to_valid_numeric(
    df["d10"],
    valid_values=[1, 2, 3],
).astype("Int64")

# 전일제 교육 종료 연령
# 실제 SAV 코드:
# 0=Refusal, 97=No full-time education, 98=Still studying, 99=DK
clean_df["education_age"] = to_valid_numeric(
    df["d8"],
    min_value=2,
    max_value=90,
)

# 교육 종료 연령이 현재 연령보다 높으면 비정상값으로 간주
clean_df.loc[
    clean_df["education_age"] > clean_df["age"],
    "education_age",
] = pd.NA

# 직업 유형
clean_df["occupation_code"] = to_valid_numeric(
    df["d15a"],
    valid_values=list(range(10, 19)),
).astype("Int64")

# 조직 규모
# 실제 SAV 코드:
# 1=1명, 2=2~9명, 3=10~49명, 4=50~250명, 5=250명 초과
# 6~9는 모름·응답거부 등으로 결측 처리
clean_df["workplace_size"] = to_valid_numeric(
    df["sd29"],
    valid_values=[1, 2, 3, 4, 5],
).astype("Int64")

# 직무 수행 디지털 역량
# 원코드: 1 Totally agree ~ 4 Totally disagree
# 역코드: 값이 높을수록 높은 디지털 역량
# 5=Not applicable, 6=Don't know은 결측 처리
digital_skill_raw = to_valid_numeric(
    df["qb2_2"],
    valid_values=[1, 2, 3, 4],
)

clean_df["digital_skill_job"] = (
    5 - digital_skill_raw
).astype("Int64")


# ============================================================
# 6. 자료 검증
# ============================================================

binary_columns = [
    "public_sector",
    "detailed_explanation",
    "personal_data_access",
    "automated_analysis_access",
    "not_informed",
    "basic_notification_only",
    "automated_monitoring_exposure",
    "automated_performance_management_exposure",
]

for column in binary_columns:
    invalid_values = clean_df.loc[
        clean_df[column].notna() & ~clean_df[column].isin([0, 1]),
        column,
    ].unique()

    if len(invalid_values) > 0:
        raise ValueError(
            f"{column}에 0/1 이외 값이 있습니다: {invalid_values}"
        )


# ============================================================
# 7. 저장
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

clean_df.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("=" * 60)
print("전처리 완료")
print(f"현재 임금근로자 필터 후: {n_after_employed_filter:,}명")
print(f"공공·비공공부문 필터 후: {n_after_sector_filter:,}명")
print(f"QB9 해당 없음·모름 제외 후: {n_final:,}명")
print(f"저장 경로: {OUTPUT_PATH}")
print("=" * 60)

print("\n[핵심 변수 결측치 수]")
print(
    clean_df[
        [
            "public_sector",
            "detailed_explanation",
            "personal_data_access",
            "automated_analysis_access",
            "not_informed",
            "age",
            "gender",
            "education_age",
            "workplace_size",
            "digital_skill_job",
        ]
    ]
    .isna()
    .sum()
    .sort_values()
)