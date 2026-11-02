import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "이 실습실에서는 뇌졸중 관련 데이터를 살펴보고, 각 열의 의미와 데이터의 특징을 확인합니다. "
    "첫 화면에서는 데이터의 전체적인 구조를 확인할 수 있습니다."
)

DATA_URL = "https://raw.githubusercontent.com/donong-20517/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_data_notes.json"

EXPECTED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi",
    "smoking_status", "stroke",
]

# ----------------------------
# 교재를 보고 입력한 기록 저장/불러오기
# ----------------------------
def load_notes():
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {"meanings": {}, "source": ""}
        data.setdefault("meanings", {})
        data.setdefault("source", "")
        return data
    except (OSError, json.JSONDecodeError):
        return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    data = {"meanings": meanings, "source": source}
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


notes = load_notes()

# ----------------------------
# 데이터 불러오기
# ----------------------------
@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


try:
    df = load_data(DATA_URL)
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

# 요청한 열 이름 순서를 우선 적용하되, 실제 파일의 추가 열은 뒤에 표시
available_expected = [column for column in EXPECTED_COLUMNS if column in df.columns]
extra_columns = [column for column in df.columns if column not in available_expected]
ordered_columns = available_expected + extra_columns
df = df[ordered_columns]

# ----------------------------
# 큰 숫자 카드 네 개
# ----------------------------
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()) if "stroke" in df.columns else 0
stroke_ratio = (stroke_count / total_people * 100) if total_people else 0

st.subheader("데이터 한눈에 보기")
card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count:,}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")

st.divider()
