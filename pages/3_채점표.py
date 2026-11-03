import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📊",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/donong-20517/data/refs/heads/main/stroke.csv"

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}


# --------------------------------------------------
# 데이터 읽기
# --------------------------------------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")

    for col in FEATURES + ["stroke"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["stroke"]).copy()
    df["stroke"] = df["stroke"].astype(int)

    return df


df = load_data(DATA_URL)


# --------------------------------------------------
# 훈련용 / 테스트용 데이터 나누기
# 10개 행마다 앞의 3개를 테스트용으로 사용
# --------------------------------------------------
test_indices = []

for start in range(0, len(df), 10):
    test_indices.extend(range(start, min(start + 3, len(df))))

test_indices = sorted(set(test_indices))

train_mask = ~df.index.isin(test_indices)

train_df = df.loc[train_mask].copy()
test_df = df.loc[~train_mask].copy()

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df["stroke"]
y_test = test_df["stroke"]

# BMI 결측값은 훈련용 데이터의 중앙값으로 채움
bmi_median = X_train["bmi"].median()

X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 나머지 결측값 처리
for col in FEATURES:
    if X_train[col].isna().any():
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

pred_logistic = logistic_model.predict(X_test)
pred_tree = tree_model.predict(X_test)


# --------------------------------------------------
# 혼동행렬과 성능지표
# --------------------------------------------------
def get_confusion_data(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return tn, fp, fn, tp


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = get_confusion_data(y_true, y_pred)

    total = tn + fp + fn + tp
    actual_positive = tp + fn
    predicted_positive = tp + fp

    accuracy = (tp + tn) / total if total else 0

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    precision = (
        tp / predicted_positive
        if predicted_positive > 0
        else None
    )

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def make_confusion_table(y_true, y_pred):
    tn, fp, fn, tp = get_confusion_data(y_true, y_pred)

    return pd.DataFrame(
        [
            [tn, fp, tn + fp],
            [fn, tp, fn + tp],
            [tn + fn, fp + tp, tn + fp + fn + tp]
        ],
        index=["실제 뇌졸중 아님", "실제 뇌졸중", "전체"],
        columns=["뇌졸중 아님 예측", "뇌졸중 예측", "행 합계"]
    )


def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value:.4f}"


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title("📊 채점표")

st.write(
    "테스트 데이터로 두 분류 모델의 예측 결과를 비교합니다. "
    "혼동행렬을 통해 실제 정답과 모델의 예측이 얼마나 일치하는지 확인합니다."
)

st.divider()

st.subheader("① 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric("테스트용 사람 수", f"{len(test_df):,}명")

with col2:
    st.metric("실제 뇌졸중인 사람 수", f"{int((y_test == 1).sum()):,}명")

st.divider()

# --------------------------------------------------
# 혼동행렬
# --------------------------------------------------
st.subheader("② 혼동행렬")

st.caption(
    "행은 실제 상태, 열은 모델의 예측입니다. "
    "TP는 뇌졸중 환자를 맞게 찾은 경우, "
    "TN은 뇌졸중이 아닌 사람을 맞게 예측한 경우입니다."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 로지스틱 회귀(확률로 답하는 모델)")
    st.dataframe(
        make_confusion_table(y_test, pred_logistic),
        width="stretch"
    )

with col2:
    st.markdown("### 의사결정트리(질문으로 답하는 모델)")
    st.dataframe(
        make_confusion_table(y_test, pred_tree),
        width="stretch"
    )


# --------------------------------------------------
# 성능지표
# --------------------------------------------------
st.divider()
st.subheader("③ 성능지표 비교")

st.caption(
    "정확도는 전체 예측 중 정답의 비율, "
    "재현율은 실제 뇌졸중 환자 중 찾아낸 비율, "
    "정밀도는 뇌졸중이라고 예측한 사람 중 실제 환자의 비율입니다."
)

baseline_value = int(y_train.mode().iloc[0])
pred_baseline = np.full(len(y_test), baseline_value)

metrics_logistic = calculate_metrics(y_test, pred_logistic)
metrics_tree = calculate_metrics(y_test, pred_tree)
metrics_baseline = calculate_metrics(y_test, pred_baseline)

comparison = pd.DataFrame({
    "성능지표": ["정확도", "재현율", "정밀도", "F1"],
    "로지스틱 회귀(확률로 답하는 모델)": [
        format_metric(metrics_logistic["정확도"]),
        format_metric(metrics_logistic["재현율"]),
        format_metric(metrics_logistic["정밀도"]),
        format_metric(metrics_logistic["F1"])
    ],
    "의사결정트리(질문으로 답하는 모델)": [
        format_metric(metrics_tree["정확도"]),
        format_metric(metrics_tree["재현율"]),
        format_metric(metrics_tree["정밀도"]),
        format_metric(metrics_tree["F1"])
    ],
    "훈련용에서 많은 쪽으로만 답하는 모델": [
        format_metric(metrics_baseline["정확도"]),
        format_metric(metrics_baseline["재현율"]),
        format_metric(metrics_baseline["정밀도"]),
        format_metric(metrics_baseline["F1"])
    ]
})

st.dataframe(comparison, hide_index=True, width="stretch")

st.info(
    "F1은 정밀도와 재현율을 함께 고려한 지표입니다. "
    "뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없습니다."
)

st.caption(
    "참고: 혼동행렬의 행 합계는 실제 인원수, 열 합계는 예측 인원수입니다."
)
