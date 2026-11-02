
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

# ----------------------------
# 열 설명 표와 우리말 뜻 입력
# ----------------------------
st.subheader("열 이름과 데이터 구조")
st.write("‘우리말 뜻’ 칸을 교재를 참고해 직접 채운 뒤, 아래 저장 버튼을 눌러 주세요.")

meanings = notes.get("meanings", {})
if not isinstance(meanings, dict):
    meanings = {}

rows = []
for column in df.columns:
    values = df[column]
    rows.append({
        "열 이름": column,
        "우리말 뜻": str(meanings.get(column, "")),
        "값의 종류": int(values.nunique(dropna=True)),
        "빈 값 개수": int(values.isna().sum()),
    })

# 편집 가능한 표. '우리말 뜻'만 수정 가능
meaning_table = pd.DataFrame(rows, columns=["열 이름", "우리말 뜻", "값의 종류", "빈 값 개수"])
edited_table = st.data_editor(
    meaning_table,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 입력하세요."),
        "값의 종류": st.column_config.NumberColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
    key="meaning_editor",
)

st.divider()

# ----------------------------
# 처음 다섯 줄
# ----------------------------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), hide_index=True, use_container_width=True)

st.divider()

# ----------------------------
# 데이터 출처 기록
# ----------------------------
st.subheader("데이터 출처 기록")
st.write("교재에 적힌 데이터 출처를 입력해 주세요. 입력한 내용은 ‘내_기록/stroke_data_notes.json’에 저장됩니다.")

source_text = st.text_area(
    "데이터 출처",
    value=str(notes.get("source", "")),
    height=100,
    placeholder="교재에 나온 데이터 출처를 여기에 적으세요.",
    key="source_editor",
)

if st.button("우리말 뜻과 출처 저장", type="primary"):
    updated_meanings = {}
    for _, row in edited_table.iterrows():
        updated_meanings[str(row["열 이름"])] = str(row["우리말 뜻"]) if pd.notna(row["우리말 뜻"]) else ""
