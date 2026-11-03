import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# ==================================================
# 1. 기본 설정
# ==================================================
st.set_page_config(
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="🚀",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/donong-20517/data/refs/heads/main/stroke.csv"

ALL_FEATURES = [
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


# ==================================================
# 2. 데이터 읽기
# ==================================================
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")

    for col in ALL_FEATURES + ["stroke"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["stroke"]).copy()
    df["stroke"] = df["stroke"].astype(int)

    return df.reset_index(drop=True)


df = load_data(DATA_URL)


# ==================================================
# 3. 입력 속성 선택
# ==================================================
st.title("🚀 성능 높이기")

st.write(
    "뇌졸중 환자는 전체 데이터에서 차지하는 비율이 매우 낮습니다. "
    "따라서 두 집단의 학습 가중치를 조정하면 예측 결과가 어떻게 달라지는지 비교합니다."
)

st.divider()

st.subheader("① 입력으로 사용할 속성 선택")

selected_features = st.multiselect(
    "모델이 학습할 속성을 선택하세요.",
    options=ALL_FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: KOREAN_NAMES[x]
)

if len(selected_features) == 0:
    st.warning("최소 한 가지 이상의 속성을 선택하세요.")
    st.stop()


# ==================================================
# 4. 훈련용 / 테스트용 데이터 나누기
# ==================================================
# 10개 행마다 앞의 3개를 테스트용으로 사용
test_indices = []

for start in range(0, len(df), 10):
    test_indices.extend(
        range(start, min(start + 3, len(df)))
    )

test_indices = sorted(set(test_indices))

train_mask = np.ones(len(df), dtype=bool)
train_mask[test_indices] = False

train_df = df.loc[train_mask].copy()
test_df = df.loc[~train_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# ==================================================
# 5. 결측값 처리
# ==================================================
# BMI는 훈련용 데이터의 중앙값으로 채움
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 선택한 다른 속성에 결측값이 있다면
# 훈련용 중앙값으로 채움
for col in selected_features:
    if X_train[col].isna().any():
        median = X_train[col].median()

        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)


# ==================================================
# 6. 모델 학습
# ==================================================
def make_logistic(class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight=class_weight
        ))
    ])


def make_tree(class_weight=None):
    return DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        class_weight=class_weight
    )


# 가중치 조정 전
logistic_before = make_logistic()
tree_before = make_tree()

# 가중치 조정 후
logistic_after = make_logistic("balanced")
tree_after = make_tree("balanced")


models = {
    "로지스틱 회귀(확률로 답하는 모델)": {
        "before": logistic_before,
        "after": logistic_after
    },
    "의사결정트리(질문으로 답하는 모델)": {
        "before": tree_before,
        "after": tree_after
    }
}

for model_group in models.values():
    model_group["before"].fit(X_train, y_train)
    model_group["after"].fit(X_train, y_train)


# 테스트 데이터 예측
predictions = {}

for name, model_group in models.items():
    predictions[name] = {
        "before": model_group["before"].predict(X_test),
        "after": model_group["after"].predict(X_test)
    }


# ==================================================
# 7. 혼동행렬과 지표 계산
# ==================================================
def get_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }


def calculate_metrics(y_true, y_pred):
    counts = get_counts(y_true, y_pred)

    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]

    total = tp + fp + fn + tn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else None
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else None
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else None
    )

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        **counts,
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_value(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value:.4f}"


def format_count(value):
    return f"{value:,}명"


# ==================================================
# 8. 테스트 데이터 정보
# ==================================================
st.divider()
st.subheader("② 테스트 데이터")

st.write(
    f"테스트용 사람 수: **{len(test_df):,}명**"
    f"　|　실제 뇌졸중인 사람 수: "
    f"**{int((y_test == 1).sum()):,}명**"
)


# ==================================================
# 9. 가중치 조정 전후 비교표
# ==================================================
st.divider()
st.subheader("③ 가중치 조정 전후 성능 비교")

st.write(
    "뇌졸중 환자와 비환자의 가중치를 맞추기 전과 후를 비교합니다."
)

for name in models:

    before = calculate_metrics(
        y_test,
        predictions[name]["before"]
    )

    after = calculate_metrics(
        y_test,
        predictions[name]["after"]
    )

    st.markdown(f"### {name}")

    comparison = pd.DataFrame({
        "항목": [
            "TP (찾아낸 환자)",
            "FN (놓친 환자)",
            "FP (헛짚은 사람)",
            "TN (정확하게 비환자로 예측)",
            "정확도",
            "재현율",
            "정밀도",
            "F1"
        ],
        "가중치 조정 전": [
            before["TP"],
            before["FN"],
            before["FP"],
            before["TN"],
            format_value(before["정확도"]),
            format_value(before["재현율"]),
            format_value(before["정밀도"]),
            format_value(before["F1"])
        ],
        "가중치 조정 후": [
            after["TP"],
            after["FN"],
            after["FP"],
            after["TN"],
            format_value(after["정확도"]),
            format_value(after["재현율"]),
            format_value(after["정밀도"]),
            format_value(after["F1"])
        ]
    })

    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        f"찾아낸 환자: {before['TP']}명 → {after['TP']}명"
        f"　|　놓친 환자: {before['FN']}명 → {after['FN']}명"
        f"　|　헛짚은 사람: {before['FP']}명 → {after['FP']}명"
    )

    st.divider()


# ==================================================
# 10. 혼동행렬
# ==================================================
st.subheader("④ 혼동행렬 비교")

st.caption(
    "행은 실제 상태, 열은 예측 결과입니다. "
    "TP는 실제 환자를 환자로 예측한 수, "
    "FP는 비환자를 환자로 예측한 수, "
    "FN은 환자를 비환자로 예측한 수, "
    "TN은 비환자를 비환자로 예측한 수입니다."
)


def confusion_table(y_true, y_pred):
    counts = get_counts(y_true, y_pred)

    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]

    return pd.DataFrame(
        [
            [
                f"TN\n{tn:,}명",
                f"FP\n{fp:,}명",
                f"{tn + fp:,}명"
            ],
            [
                f"FN\n{fn:,}명",
                f"TP\n{tp:,}명",
                f"{fn + tp:,}명"
            ],
            [
                f"{tn + fn:,}명",
                f"{fp + tp:,}명",
                f"{tn + fp + fn + tp:,}명"
            ]
        ],
        index=[
            "실제 뇌졸중 아님",
            "실제 뇌졸중",
            "열 합계"
        ],
        columns=[
            "뇌졸중 아님 예측",
            "뇌졸중 예측",
            "전체"
        ]
    )


for name in models:

    st.markdown(f"### {name}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**가중치 조정 전**")

        st.dataframe(
            confusion_table(
                y_test,
                predictions[name]["before"]
            ),
            width="stretch"
        )

    with col2:
        st.markdown("**가중치 조정 후**")

        st.dataframe(
            confusion_table(
                y_test,
                predictions[name]["after"]
            ),
            width="stretch"
        )


# ==================================================
# 11. 놓친 환자 / 헛짚은 사람 목록
# ==================================================
st.divider()
st.subheader("⑤ 잘못 예측한 사람 확인")

model_choice = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    list(models.keys())
)

weight_choice = st.radio(
    "가중치 조정 상태",
    ["조정 전", "조정 후"],
    horizontal=True
)

state = "before" if weight_choice == "조정 전" else "after"

selected_prediction = predictions[model_choice][state]


def make_person_list(mask):
    result = test_df.loc[mask].copy()

    result["번호"] = result.index + 1

    result["고혈압 여부"] = result["hypertension"].map({
        0: "아니오",
        1: "예"
    })

    result["심장병 여부"] = result["heart_disease"].map({
        0: "아니오",
        1: "예"
    })

    result = result.rename(columns={
        "age": "나이",
        "avg_glucose_level": "평균 혈당"
    })

    result = result.sort_values(
        "나이",
        ascending=False
    )

    columns = [
        "번호",
        "나이",
        "평균 혈당",
        "고혈압 여부",
        "심장병 여부"
    ]

    return result[columns]


missed_mask = (
    (y_test.to_numpy() == 1)
    & (selected_prediction == 0)
)

false_alarm_mask = (
    (y_test.to_numpy() == 0)
    & (selected_prediction == 1)
)

missed_people = make_person_list(missed_mask)
false_alarm_people = make_person_list(false_alarm_mask)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"### 놓친 환자 ({len(missed_people):,}명)"
    )

    st.caption(
        "실제 뇌졸중 환자이지만 뇌졸중이 아니라고 예측한 사람"
    )

    st.dataframe(
        missed_people.head(20),
        hide_index=True,
        width="stretch"
    )

    if len(missed_people) > 20:
        st.caption("20명까지만 표시했습니다.")

with col2:
    st.markdown(
        f"### 헛짚은 사람 ({len(false_alarm_people):,}명)"
    )

    st.caption(
        "실제 뇌졸중 환자가 아니지만 뇌졸중이라고 예측한 사람"
    )

    st.dataframe(
        false_alarm_people.head(20),
        hide_index=True,
        width="stretch"
    )

    if len(false_alarm_people) > 20:
        st.caption("20명까지만 표시했습니다.")


st.info(
    f"목록 인원 확인: 놓친 환자 {len(missed_people)}명 = "
    f"혼동행렬의 FN {get_counts(y_test, selected_prediction)['FN']}명"
    f"　|　헛짚은 사람 {len(false_alarm_people)}명 = "
    f"혼동행렬의 FP {get_counts(y_test, selected_prediction)['FP']}명"
)


# ==================================================
# 12. Plotly 2차원 / 3차원 산점도
# ==================================================
st.divider()
st.subheader("⑥ 실제 뇌졸중 여부와 0.5 경계 비교")

st.write(
    "점의 색은 실제 뇌졸중 여부를 나타냅니다. "
    "서로 다른 색의 경계는 가중치 조정 전후 모델의 "
    "뇌졸중 예측 확률이 0.5가 되는 지점을 나타냅니다."
)

if len(selected_features) < 2:

    st.warning(
        "산점도를 그리려면 입력 속성을 두 가지 이상 선택해야 합니다."
    )

else:

    plot_model_choice = st.selectbox(
        "경계선을 확인할 모델",
        list(models.keys()),
        key="plot_model_choice"
    )

    before_model = models[plot_model_choice]["before"]
    after_model = models[plot_model_choice]["after"]

    # 나머지 속성은 훈련용 중앙값으로 고정
    feature_medians = X_train.median()

    # 실제 데이터 표시용
    plot_data = X_test.copy()
    plot_data["stroke"] = y_test.to_numpy()

    # --------------------------------------------------
    # 2차원 산점도
    # --------------------------------------------------
    if len(selected_features) == 2:

        x_feature = selected_features[0]
        y_feature = selected_features[1]

        fig = go.Figure()

        # 실제 뇌졸중이 아닌 사람
        for label, color, text_label in [
            (0, "blue", "실제 뇌졸중 아님"),
            (1, "red", "실제 뇌졸중")
        ]:

            subset = plot_data[
                plot_data["stroke"] == label
            ]

            fig.add_trace(go.Scatter(
                x=subset[x_feature],
                y=subset[y_feature],
                mode="markers",
                name=text_label,
                marker=dict(
                    color=color,
                    size=7,
                    opacity=0.65
                )
            ))

        # 경계선 계산용 격자
        x_min, x_max = X_train[x_feature].min(), X_train[x_feature].max()
        y_min, y_max = X_train[y_feature].min(), X_train[y_feature].max()

        x_values = np.linspace(x_min, x_max, 100)
        y_values = np.linspace(y_min, y_max, 100)

        xx, yy = np.meshgrid(x_values, y_values)

        grid = pd.DataFrame({
            col: np.full(xx.size, feature_medians[col])
            for col in selected_features
        })

        grid[x_feature] = xx.ravel()
        grid[y_feature] = yy.ravel()

        before_prob = before_model.predict_proba(grid)[:, 1]
        after_prob = after_model.predict_proba(grid)[:, 1]

        before_z = before_prob.reshape(xx.shape)
        after_z = after_prob.reshape(xx.shape)

        # 0.5 경계선
        fig.add_trace(go.Contour(
            x=x_values,
            y=y_values,
            z=before_z,
            name="가중치 조정 전 0.5 경계",
            showscale=False,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="lines"
            ),
            line=dict(
                color="green",
                width=3,
                dash="solid"
            ),
            hoverinfo="skip"
        ))

        fig.add_trace(go.Contour(
            x=x_values,
            y=y_values,
            z=after_z,
            name="가중치 조정 후 0.5 경계",
            showscale=False,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="lines"
            ),
            line=dict(
                color="orange",
                width=3,
                dash="dash"
            ),
            hoverinfo="skip"
        ))

        fig.update_layout(
            title="2차원 산점도와 0.5 경계선",
            xaxis_title=KOREAN_NAMES[x_feature],
            yaxis_title=KOREAN_NAMES[y_feature],
            height=650,
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        before_outside = (
            before_z.min() > 0.5
            or before_z.max() < 0.5
        )

        after_outside = (
            after_z.min() > 0.5
            or after_z.max() < 0.5
        )

        if before_outside:
            st.write(
                "가중치 조정 전: 0.5 경계선이 현재 그림 범위 밖에 있습니다."
            )
        else:
            st.write(
                "가중치 조정 전: 0.5 경계선이 그림 안에 있습니다."
            )

        if after_outside:
            st.write(
                "가중치 조정 후: 0.5 경계선이 현재 그림 범위 밖에 있습니다."
            )
        else:
            st.write(
                "가중치 조정 후: 0.5 경계선이 그림 안에 있습니다."
            )

    # --------------------------------------------------
    # 3차원 산점도
    # --------------------------------------------------
    else:

        x_feature = selected_features[0]
        y_feature = selected_features[1]
        z_feature = selected_features[2]

        fig = go.Figure()

        for label, color, text_label in [
            (0, "blue", "실제 뇌졸중 아님"),
            (1, "red", "실제 뇌졸중")
        ]:

            subset = plot_data[
                plot_data["stroke"] == label
            ]

            fig.add_trace(go.Scatter3d(
                x=subset[x_feature],
                y=subset[y_feature],
                z=subset[z_feature],
                mode="markers",
                name=text_label,
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.7
                )
            ))

        # 나머지 선택 속성은 중앙값으로 고정
        x_values = np.linspace(
            X_train[x_feature].min(),
            X_train[x_feature].max(),
            30
        )

        y_values = np.linspace(
            X_train[y_feature].min(),
            X_train[y_feature].max(),
            30
        )

        z_values = np.linspace(
            X_train[z_feature].min(),
            X_train[z_feature].max(),
            100
        )

        xx, yy = np.meshgrid(x_values, y_values)

        def calculate_boundary_surface(model):

            boundary_z = np.full(xx.shape, np.nan)

            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):

                    grid = pd.DataFrame({
                        col: [feature_medians[col]] * len(z_values)
                        for col in selected_features
                    })

                    grid[x_feature] = xx[i, j]
                    grid[y_feature] = yy[i, j]
                    grid[z_feature] = z_values

                    probabilities = model.predict_proba(grid)[:, 1]

                    differences = probabilities - 0.5

                    crossing = np.where(
                        differences[:-1] * differences[1:] <= 0
                    )[0]

                    if len(crossing) > 0:

                        k = crossing[0]

                        p1 = probabilities[k]
                        p2 = probabilities[k + 1]

                        if p2 != p1:
                            ratio = (0.5 - p1) / (p2 - p1)
                        else:
                            ratio = 0

                        boundary_z[i, j] = (
                            z_values[k]
                            + ratio * (z_values[k + 1] - z_values[k])
                        )

            return boundary_z

        before_surface = calculate_boundary_surface(before_model)
        after_surface = calculate_boundary_surface(after_model)

        before_exists = np.isfinite(before_surface).any()
        after_exists = np.isfinite(after_surface).any()

        if before_exists:
            fig.add_trace(go.Surface(
                x=xx,
                y=yy,
                z=before_surface,
                name="가중치 조정 전 0.5 평면",
                opacity=0.45,
                showscale=False,
                colorscale=[
                    [0, "green"],
                    [1, "green"]
                ],
                connectgaps=False
            ))

        if after_exists:
            fig.add_trace(go.Surface(
                x=xx,
                y=yy,
                z=after_surface,
                name="가중치 조정 후 0.5 평면",
                opacity=0.45,
                showscale=False,
                colorscale=[
                    [0, "orange"],
                    [1, "orange"]
                ],
                connectgaps=False
            ))

        fig.update_layout(
            title="3차원 산점도와 0.5 경계 평면",
            scene=dict(
                xaxis_title=KOREAN_NAMES[x_feature],
                yaxis_title=KOREAN_NAMES[y_feature],
                zaxis_title=KOREAN_NAMES[z_feature]
            ),
            height=750,
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        if not before_exists:
            st.write(
                "가중치 조정 전: 0.5 경계 평면이 현재 그림 범위 밖에 있습니다."
            )
        else:
            st.write(
                "가중치 조정 전: 0.5 경계 평면을 표시했습니다."
            )

        if not after_exists:
            st.write(
                "가중치 조정 후: 0.5 경계 평면이 현재 그림 범위 밖에 있습니다."
            )
        else:
            st.write(
                "가중치 조정 후: 0.5 경계 평면을 표시했습니다."
            )


# ==================================================
# 13. 마무리 안내
# ==================================================
st.divider()

st.subheader("⑦ 결과 해석")

st.write(
    """
    - 재현율이 높아지면 실제 뇌졸중 환자를 더 많이 찾아낸 것입니다.
    - 정밀도가 낮아지면 뇌졸중이 아닌 사람을 환자로 잘못 예측하는 경우가 늘어날 수 있습니다.
    - 정확도 하나만으로 모델의 성능을 판단하기 어렵습니다.
    - 뇌졸중처럼 한쪽 집단의 수가 매우 적은 데이터에서는 재현율과 정밀도, F1도 함께 확인해야 합니다.
    """
)

st.warning(
    "이 페이지의 모델은 데이터 분석과 분류 학습을 위한 실습용입니다. "
    "실제 환자의 뇌졸중 진단이나 의료적 판단에 사용할 수 없습니다."
)
