import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="데이터 탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

# 기존 앱에서 사용하는 데이터 주소
DATA_URL = "https://raw.githubusercontent.com/donong-20517/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.write(
    "뇌졸중 데이터의 분포와 그룹별 차이를 그래프로 살펴봅니다. "
    "그래프에서 발견한 차이만으로 원인과 결과를 단정할 수는 없습니다."
)

# 데이터 불러오기
try:
    df = load_data(DATA_URL)
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 데이터 주소와 인터넷 연결을 확인해 주세요.")
    st.code(str(error))
    st.stop()


# 필요한 열 확인
required_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()


# 숫자형 데이터로 변환
for col in [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 1. 나이와 평균 혈당 분포
# --------------------------------------------------
st.header("1. 나이와 평균 혈당 분포")

col1, col2 = st.columns(2)

with col1:
    age_data = df.dropna(subset=["age"])

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    glucose_data = df.dropna(subset=["avg_glucose_level"])

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)

st.info(
    "이 그래프로 알 수 있는 것: 나이와 평균 혈당이 어떤 범위에 많이 분포하는지 확인할 수 있습니다."
)


# --------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이와 혈당 비교
# --------------------------------------------------
st.header("2. 뇌졸중 여부에 따른 비교")

df["뇌졸중 여부"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

group_order = ["뇌졸중 없음", "뇌졸중 있음"]

col1, col2 = st.columns(2)

with col1:
    age_box_data = df.dropna(subset=["age", "뇌졸중 여부"])

    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": group_order},
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹"
        }
    )

    fig_age_box.update_layout(showlegend=False)
    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    glucose_box_data = df.dropna(
        subset=["avg_glucose_level", "뇌졸중 여부"]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": group_order},
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹"
        }
    )

    fig_glucose_box.update_layout(showlegend=False)
    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 그룹별 평균값 표
mean_rows = []

for stroke_value, group_name in [
    (0, "뇌졸중 없음"),
    (1, "뇌졸중 있음")
]:
    group = df[df["stroke"] == stroke_value]

    mean_rows.append({
        "그룹": group_name,
        "사람 수": len(group),
        "평균 나이": round(group["age"].mean(), 2),
        "평균 혈당": round(group["avg_glucose_level"].mean(), 2)
    })

mean_table = pd.DataFrame(mean_rows)

st.subheader("그룹별 평균 비교")
st.dataframe(mean_table, hide_index=True, use_container_width=True)

st.info(
    "이 그래프로 알 수 있는 것: 뇌졸중 발생 여부에 따라 나이와 평균 혈당의 분포 및 평균이 어떻게 다른지 비교할 수 있습니다."
)


# --------------------------------------------------
# 3. 고혈압과 심장병 여부에 따른 뇌졸중 비율
# --------------------------------------------------
st.header("3. 고혈압과 심장병 여부에 따른 뇌졸중 비율")


def make_rate_chart(data, column, labels, title):
    chart_data = data[
        data[column].isin([0, 1]) &
        data["stroke"].isin([0, 1])
    ].copy()

    chart_data["구분"] = chart_data[column].map(labels)

    summary = (
        chart_data.groupby("구분", as_index=False)
        .agg(
            사람 수=("stroke", "size"),
            뇌졸중_비율=("stroke", "mean")
        )
    )

    summary["뇌졸중 비율(%)"] = summary["뇌졸중_비율"] * 100
    summary["표시값"] = summary["뇌졸중 비율(%)"].map(
        lambda x: f"{x:.2f}%"
    )

    fig = px.bar(
        summary,
        x="구분",
        y="뇌졸중 비율(%)",
        text="표시값",
        title=title,
        labels={
            "구분": "구분",
            "뇌졸중 비율(%)": "뇌졸중 비율(%)"
        }
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_rangemode="tozero")

    return fig, summary


col1, col2 = st.columns(2)

with col1:
    hypertension_fig, hypertension_summary = make_rate_chart(
        df,
        "hypertension",
        {
            0: "고혈압 없음",
            1: "고혈압 있음"
        },
        "고혈압 여부에 따른 뇌졸중 비율"
    )

    st.plotly_chart(hypertension_fig, use_container_width=True)

    st.dataframe(
        hypertension_summary[
            ["구분", "사람 수", "뇌졸중 비율(%)"]
        ].round(2),
        hide_index=True,
        use_container_width=True
    )

with col2:
    heart_fig, heart_summary = make_rate_chart(
        df,
        "heart_disease",
        {
            0: "심장병 없음",
            1: "심장병 있음"
        },
        "심장병 여부에 따른 뇌졸중 비율"
    )

    st.plotly_chart(heart_fig, use_container_width=True)

    st.dataframe(
        heart_summary[
            ["구분", "사람 수", "뇌졸중 비율(%)"]
        ].round(2),
        hide_index=True,
        use_container_width=True
    )

st.info(
    "이 그래프로 알 수 있는 것: 고혈압 또는 심장병 여부에 따라 뇌졸중 발생 비율이 어떻게 다른지 비교할 수 있습니다. "
    "이 차이만으로 해당 질환이 뇌졸중의 직접적인 원인이라고 결론 내릴 수는 없습니다."
)


# --------------------------------------------------
# 4. BMI 결측 여부와 뇌졸중 비율
# --------------------------------------------------
st.header("4. BMI 결측 여부와 뇌졸중 비율")

valid_stroke = df[df["stroke"].isin([0, 1])].copy()

overall_count = len(valid_stroke)
overall_strokes = int((valid_stroke["stroke"] == 1).sum())

overall_rate = (
    overall_strokes / overall_count * 100
    if overall_count > 0
    else float("nan")
)

missing_bmi = valid_stroke[valid_stroke["bmi"].isna()]

missing_count = len(missing_bmi)
missing_strokes = int((missing_bmi["stroke"] == 1).sum())

missing_rate = (
    missing_strokes / missing_count * 100
    if missing_count > 0
    else float("nan")
)

bmi_table = pd.DataFrame([
    {
        "대상": "BMI 결측자",
        "사람 수": missing_count,
        "뇌졸중 인원": missing_strokes,
        "뇌졸중 비율(%)": round(missing_rate, 2)
    },
    {
        "대상": "전체 데이터",
        "사람 수": overall_count,
        "뇌졸중 인원": overall_strokes,
        "뇌졸중 비율(%)": round(overall_rate, 2)
    }
])

st.dataframe(
    bmi_table,
    hide_index=True,
    use_container_width=True
)

st.info(
    "이 표로 알 수 있는 것: BMI 값이 비어 있는 사람들의 뇌졸중 비율과 전체 데이터의 비율을 비교할 수 있습니다."
)


# --------------------------------------------------
# 5. 흡연 상태별 인원수
# --------------------------------------------------
st.header("5. 흡연 상태별 인원수")

smoking_data = df.copy()
smoking_data["smoking_status"] = (
    smoking_data["smoking_status"]
    .fillna("결측값")
    .replace("", "결측값")
)

smoking_summary = (
    smoking_data["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_summary,
    hide_index=True,
    use_container_width=True
)

smoking_fig = px.bar(
    smoking_summary,
    x="흡연 상태",
    y="사람 수",
    text="사람 수",
    title="흡연 상태별 인원수",
    labels={
        "흡연 상태": "흡연 상태",
        "사람 수": "사람 수"
    }
)

smoking_fig.update_traces(textposition="outside")
smoking_fig.update_layout(yaxis_rangemode="tozero")

st.plotly_chart(smoking_fig, use_container_width=True)

st.info(
    "이 그래프로 알 수 있는 것: 데이터에 기록된 흡연 상태별 인원수와 결측값의 규모를 확인할 수 있습니다."
)
