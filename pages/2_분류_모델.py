import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ==================================================
# 페이지 설정
# ==================================================

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/donong-20517/data/refs/heads/main/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

ALL_FEATURES = list(FEATURE_NAMES.keys())


# ==================================================
# 데이터 불러오기
# ==================================================

@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


st.title("🤖 분류 모델")
st.write(
    "여러 속성을 이용해 뇌졸중 발생 여부를 예측하는 "
    "분류 모델을 만들고 성능을 비교합니다."
)

try:
    df = load_data(DATA_URL)

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


# 필요한 열 확인
required_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# 숫자형으로 변환
for col in required_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# 번호와 정답이 없는 행 제외
df = df.dropna(
    subset=["id", "stroke"]
).copy()

df = df[df["stroke"].isin([0, 1])].copy()

df["stroke"] = df["stroke"].astype(int)

# 번호 순서로 정렬
df = df.sort_values("id").reset_index(drop=True)


# ==================================================
# 1. 입력 속성 선택
# ==================================================

st.header("1. 입력 속성 선택")

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=ALL_FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("입력 속성은 두 개 이상 선택해야 합니다.")
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(
        FEATURE_NAMES[x]
        for x in selected_features
    )
)


# ==================================================
# 2. 훈련 데이터와 테스트 데이터 분리
# ==================================================

st.header("2. 데이터 분리")

# 번호 순으로 정렬된 상태에서
# 10명씩 묶고 각 묶음의 앞 3명을 테스트용으로 지정
row_positions = np.arange(len(df))

test_mask = row_positions % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

st.write(
    f"전체 데이터: **{len(df):,}명**"
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련 데이터",
        f"{len(train_df):,}명"
    )

with col2:
    st.metric(
        "테스트 데이터",
        f"{len(test_df):,}명"
    )

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르므로 원본 데이터의 행 수를 확인해 주세요."
    )


# 입력 데이터와 정답 분리
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# ==================================================
# 3. 결측값 처리
# ==================================================

st.header("3. 결측값 처리")

# BMI를 선택한 경우에만 훈련 데이터 중앙값으로 채우기
if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련 데이터에서 체질량지수의 중앙값을 계산할 수 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )

    st.info(
        f"체질량지수의 결측값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )

# BMI 이외의 선택 속성에 결측값이 있는지 확인
other_features = [
    col for col in selected_features
    if col != "bmi"
]

missing_train = X_train[other_features].isna().sum()
missing_test = X_test[other_features].isna().sum()

if missing_train.sum() > 0 or missing_test.sum() > 0:
    st.error(
        "선택한 속성 중 BMI 이외의 열에 결측값이 있습니다. "
        "해당 데이터를 확인해 주세요."
    )
    st.stop()


# ==================================================
# 4. 모델 만들기
# ==================================================

st.header("4. 분류 모델 학습")

# 로지스틱 회귀는 훈련 데이터의 평균과 표준편차로 크기 조정
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 로지스틱 회귀
# --------------------------------------------------

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 의사결정트리
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 아무 속성도 보지 않는 기준 모델
# --------------------------------------------------

# 훈련 데이터에서 더 많은 쪽을 항상 답하는 모델
majority_class = y_train.value_counts().idxmax()

baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)


# ==================================================
# 5. 정확도 계산
# ==================================================

# 로지스틱 회귀 예측
logistic_train_pred = logistic_model.predict(
    X_train_scaled
)

logistic_test_pred = logistic_model.predict(
    X_test_scaled
)

# 의사결정트리 예측
tree_train_pred = tree_model.predict(
    X_train
)

tree_test_pred = tree_model.predict(
    X_test
)


# 정확도
logistic_train_acc = accuracy_score(
    y_train,
    logistic_train_pred
)

logistic_test_acc = accuracy_score(
    y_test,
    logistic_test_pred
)

tree_train_acc = accuracy_score(
    y_train,
    tree_train_pred
)

tree_test_acc = accuracy_score(
    y_test,
    tree_test_pred
)

baseline_train_acc = accuracy_score(
    y_train,
    baseline_train_pred
)

baseline_test_acc = accuracy_score(
    y_test,
    baseline_test_pred
)


# ==================================================
# 6. 정확도 카드
# ==================================================

st.header("5. 모델별 정확도 비교")

col1, col2, col3 = st.columns(3)


def accuracy_card(
    container,
    title,
    train_acc,
    test_acc
):
    with container:
        st.subheader(title)

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_acc * 100:.2f}%"
        )

        small1, small2 = st.columns(2)

        with small1:
            st.caption("훈련 데이터")
            st.write(f"**{train_acc * 100:.2f}%**")

        with small2:
            st.caption("테스트 데이터")
            st.write(f"**{test_acc * 100:.2f}%**")


accuracy_card(
    col1,
    "로지스틱 회귀(확률로 답하는 모델)",
    logistic_train_acc,
    logistic_test_acc
)

accuracy_card(
    col2,
    "의사결정트리(질문으로 답하는 모델)",
    tree_train_acc,
    tree_test_acc
)

accuracy_card(
    col3,
    "기준 모델(다수 클래스만 답하는 모델)",
    baseline_train_acc,
    baseline_test_acc
)


st.info(
    "정확도는 전체 예측 중 실제 정답과 일치한 예측의 비율입니다. "
    "테스트 데이터 정확도를 중심으로 모델을 비교하세요."
)


# ==================================================
# 7. 산점도와 모델의 영역
# ==================================================

st.header("6. 두 속성으로 살펴보는 분류 결과")

st.write(
    "선택한 속성 중 두 개를 골라 가로축과 세로축으로 사용합니다. "
    "나머지 속성은 테스트 데이터의 중앙값에 고정하여 계산합니다."
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with axis_col2:
    y_feature_options = [
        x for x in selected_features
        if x != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_feature_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )


# 나머지 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}

for feature in selected_features:

    if feature not in [x_feature, y_feature]:

        fixed_value = X_test[feature].median()

        fixed_values[feature] = fixed_value


if fixed_values:

    fixed_text = []

    for feature, value in fixed_values.items():
        fixed_text.append(
            f"{FEATURE_NAMES[feature]}: {value:.2f}"
        )

    st.write(
        "**고정한 속성의 값:** "
        + " / ".join(fixed_text)
    )

else:
    st.write(
        "선택한 속성이 두 개뿐이므로 고정한 속성은 없습니다."
    )


# --------------------------------------------------
# 그래프용 격자 만들기
# --------------------------------------------------

x_min = X_test[x_feature].min()
x_max = X_test[x_feature].max()

y_min = X_test[y_feature].min()
y_max = X_test[y_feature].max()

x_padding = (x_max - x_min) * 0.08
y_padding = (y_max - y_min) * 0.08

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

x_grid = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    120
)

y_grid = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    120
)

xx, yy = np.meshgrid(
    x_grid,
    y_grid
)


# 격자 데이터 만들기
grid_data = pd.DataFrame(
    index=np.arange(xx.size)
)

for feature in selected_features:

    if feature == x_feature:
        grid_data[feature] = xx.ravel()

    elif feature == y_feature:
        grid_data[feature] = yy.ravel()

    else:
        grid_data[feature] = fixed_values[feature]


# BMI가 선택된 경우 훈련 중앙값으로 결측값 처리
if "bmi" in selected_features:
    grid_data["bmi"] = grid_data["bmi"].fillna(
        bmi_median
    )


# --------------------------------------------------
# 로지스틱 회귀의 확률 계산
# --------------------------------------------------

grid_scaled = scaler.transform(
    grid_data[selected_features]
)

grid_probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1]

probability_grid = grid_probability.reshape(
    xx.shape
)


# --------------------------------------------------
# 의사결정트리의 영역 계산
# --------------------------------------------------

tree_grid_prediction = tree_model.predict(
    grid_data[selected_features]
)

tree_prediction_grid = tree_grid_prediction.reshape(
    xx.shape
)


# --------------------------------------------------
# 테스트 데이터 산점도
# --------------------------------------------------

plot_data = X_test.copy()

plot_data["실제 결과"] = y_test.map({
    0: "뇌졸중 아님",
    1: "뇌졸중"
})

plot_data["실제 결과"] = pd.Categorical(
    plot_data["실제 결과"],
    categories=["뇌졸중 아님", "뇌졸중"]
)


# --------------------------------------------------
# Plotly 그림
# --------------------------------------------------

fig = go.Figure()


# 의사결정트리 영역을 옅은 색으로 표시
fig.add_trace(
    go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=tree_prediction_grid,
        colorscale=[
            [0, "rgba(100, 149, 237, 0.25)"],
            [1, "rgba(255, 99, 71, 0.25)"]
        ],
        showscale=False,
        opacity=0.35,
        hoverinfo="skip",
        name="의사결정트리 영역"
    )
)


# 로지스틱 회귀의 0.5 경계선
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=probability_grid,
        contours={
            "start": 0.5,
            "end": 0.5,
            "size": 1,
            "coloring": "lines",
            "showlabels": True
        },
        line={
            "color": "black",
            "width": 3
        },
        showscale=False,
        name="로지스틱 회귀 0.5 경계선",
        hoverinfo="skip"
    )
)


# 실제 테스트 데이터 점 표시
for label, color in [
    ("뇌졸중 아님", "blue"),
    ("뇌졸중", "red")
]:

    selected_plot_data = plot_data[
        plot_data["실제 결과"] == label
    ]

    fig.add_trace(
        go.Scatter(
            x=selected_plot_data[x_feature],
            y=selected_plot_data[y_feature],
            mode="markers",
            name=label,
            marker={
                "color": color,
                "size": 7,
                "opacity": 0.7
            },
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y}}"
                "<extra>%{fullData.name}</extra>"
            )
        )
    )


fig.update_layout(
    title="테스트 데이터와 분류 모델의 경계",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="실제 뇌졸중 여부",
    height=650
)

st.plotly_chart(
    fig,
    width="stretch"
)


# --------------------------------------------------
# 경계선이 그림 안에 있는지 확인
# --------------------------------------------------

boundary_exists = (
    probability_grid.min() <= 0.5
    and probability_grid.max() >= 0.5
)

if boundary_exists:
    st.write(
        "로지스틱 회귀의 0.5 경계선이 그림의 범위 안에 있습니다."
    )

else:
    st.write(
        "로지스틱 회귀의 0.5 경계선은 현재 그림의 범위 밖에 있습니다."
    )


st.caption(
    "산점도의 점 색은 실제 뇌졸중 여부를 나타냅니다. "
    "검은 선은 로지스틱 회귀 모델의 예측 확률이 0.5인 경계이며, "
    "옅은 배경색은 의사결정트리 모델의 예측 영역입니다."
)


# ==================================================
# 8. 의사결정트리 가지 그림
# ==================================================

st.header("7. 의사결정트리가 던진 질문")

st.write(
    "의사결정트리는 속성의 기준값을 질문하며 데이터를 나눕니다. "
    "아래 그림은 훈련 데이터로 학습한 의사결정트리입니다."
)


# --------------------------------------------------
# 각 노드에 도달한 훈련 데이터 계산
# --------------------------------------------------

decision_path = tree_model.decision_path(
    X_train
)

tree_structure = tree_model.tree_

node_statistics = {}

for node_id in range(tree_structure.node_count):

    reached = decision_path[:, node_id].toarray().ravel().astype(bool)

    node_y = y_train.to_numpy()[reached]

    total_count = len(node_y)
    positive_count = int(node_y.sum())

    positive_rate = (
        positive_count / total_count * 100
        if total_count > 0
        else 0
    )

    node_statistics[node_id] = {
        "total": total_count,
        "positive": positive_count,
        "rate": positive_rate
    }


# --------------------------------------------------
# DOT 문자열 만들기
# --------------------------------------------------

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, nodesep=0.45, ranksep=0.6];',
    'node [shape=box, style="filled,rounded", fontname="Malgun Gothic", fontsize=10];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]


def escape_dot(text):
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


for node_id in range(tree_structure.node_count):

    stats = node_statistics[node_id]

    total = stats["total"]
    positive = stats["positive"]
    rate = stats["rate"]

    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )

    if is_leaf:

        # 말단 노드가 예측하는 결과
        leaf_class = int(
            tree_model.predict(
                X_train.iloc[
                    np.where(
                        decision_path[:, node_id].toarray().ravel()
                    )[0]
                ].iloc[:1]
            )[0]
        )

        if leaf_class == 1:
            fill_color = "#F8B4B4"
            answer_text = "답: 뇌졸중"

        else:
            fill_color = "#B8D5F5"
            answer_text = "답: 뇌졸중 아님"

        node_label = (
            f"말단 노드 {node_id}\\n"
            f"훈련용 사람 수: {total}명\\n"
            f"뇌졸중인 사람: {positive}명\\n"
            f"뇌졸중 비율: {rate:.1f}%\\n"
            f"{answer_text}"
        )

    else:

        feature_index = tree_structure.feature[node_id]
        feature = selected_features[feature_index]

        threshold = tree_structure.threshold[node_id]

        fill_color = "#FFF2CC"

        node_label = (
            f"질문 {node_id}\\n"
            f"{FEATURE_NAMES[feature]} <= {threshold:.2f}?\\n"
            f"훈련용 사람 수: {total}명\\n"
            f"뇌졸중인 사람: {positive}명\\n"
            f"뇌졸중 비율: {rate:.1f}%"
        )

    dot_lines.append(
        f'n{node_id} [label="{escape_dot(node_label)}", fillcolor="{fill_color}"];'
    )


# 가지 연결
for node_id in range(tree_structure.node_count):

    left_child = tree_structure.children_left[node_id]
    right_child = tree_structure.children_right[node_id]

    is_leaf = left_child == right_child

    if not is_leaf:

        # 왼쪽 가지: 조건이 참
        dot_lines.append(
            f'n{node_id} -> n{left_child} [label="예"];'
        )

        # 오른쪽 가지: 조건이 거짓
        dot_lines.append(
            f'n{node_id} -> n{right_child} [label="아니요"];'
        )


dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    width="stretch"
)


# ==================================================
# 9. 의사결정트리 결과 정리
# ==================================================

leaf_nodes = [
    node_id
    for node_id in range(tree_structure.node_count)
    if (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )
]

negative_leaf_count = 0

for node_id in leaf_nodes:

    stats = node_statistics[node_id]

    # 해당 말단 노드에 도달한 훈련 데이터의 다수 클래스
    if stats["total"] > 0:

        negative_count = (
            stats["total"] - stats["positive"]
        )

        if negative_count >= stats["positive"]:
            negative_leaf_count += 1


# 실제 질문에 사용된 속성
used_feature_indices = sorted(
    set(
        tree_structure.feature[
            tree_structure.feature >= 0
        ]
    )
)

used_features = [
    FEATURE_NAMES[selected_features[index]]
    for index in used_feature_indices
]


st.subheader("의사결정트리 요약")

st.write(
    f"- 답을 내리는 마디(말단 노드)는 모두 **{len(leaf_nodes)}칸**입니다."
)

st.write(
    f"- 그중 뇌졸중 아님이라고 답하는 마디는 **{negative_leaf_count}칸**입니다."
)

if used_features:

    st.write(
        "- 실제로 질문에 사용한 속성: "
        + ", ".join(used_features)
    )

else:

    st.write(
        "- 실제로 질문에 사용한 속성: 없음"
    )


st.info(
    "의사결정트리는 최대 세 번까지만 질문하도록 설정했습니다. "
    "또한 말단 노드에 훈련 데이터가 다섯 명 미만으로 남지 않도록 설정했습니다."
)


# ==================================================
# 10. 마무리
# ==================================================

st.divider()

st.caption(
    "주의: 이 앱은 학습용 데이터로 분류 모델의 작동 원리를 살펴보는 실습입니다. "
    "실제 의료 진단이나 개인의 뇌졸중 위험 판단에 사용해서는 안 됩니다."
)
