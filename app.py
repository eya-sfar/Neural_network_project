import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from models.mlp import MLP
from models.perceptron import Perceptron, PerceptronActivation
from training.trainer import Trainer
from utils.preprocessing import prepare_data
from utils.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    mse,
    rmse,
    r2_score
)


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="NeuroLearn",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0d0f14;
    color: #e8eaf0;
}

[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #e8eaf0;
}

.main .block-container {
    padding-top: 1.2rem;
    max-width: 100%;
}

.section-title {
    font-size: 11px;
    font-weight: 700;
    color: #8890a4;
    text-transform: uppercase;
    margin-top: 18px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 6px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1a1d27;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 6px 13px;
    border-radius: 20px;
    font-size: 13px;
}

.dot-green {
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    display: inline-block;
}

.dot-gray {
    width: 8px;
    height: 8px;
    background: #545c70;
    border-radius: 50%;
    display: inline-block;
}

.info-box {
    background: #1a1d27;
    border: 1px solid rgba(124,106,247,0.25);
    border-left: 3px solid #7c6af7;
    padding: 11px;
    border-radius: 8px;
    font-size: 13px;
    color: #c8cbd4;
    margin-bottom: 10px;
}

.warn-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-left: 3px solid #fbbf24;
    padding: 11px;
    border-radius: 8px;
    font-size: 13px;
    color: #fbbf24;
    margin-bottom: 10px;
}

.error-box {
    background: rgba(249,120,120,0.08);
    border: 1px solid rgba(249,120,120,0.25);
    border-left: 3px solid #f97878;
    padding: 11px;
    border-radius: 8px;
    font-size: 13px;
    color: #f97878;
    margin-bottom: 10px;
}

[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 15px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Constants
# ============================================================

COLORS = {
    "purple": "#7c6af7",
    "teal": "#2dd4bf",
    "coral": "#f97878",
    "amber": "#fbbf24",
    "green": "#34d399",
    "bg": "#0d0f14",
    "bg2": "#13161e",
    "text": "#e8eaf0",
    "text2": "#8890a4"
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(19,22,30,0.75)",
    font=dict(color=COLORS["text2"], size=11),
    margin=dict(l=45, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)")
)


# ============================================================
# Helper functions
# ============================================================

def safe_parse_hidden_layers(text):
    try:
        values = [
            int(x.strip())
            for x in text.split(",")
            if x.strip() != ""
        ]

        if len(values) == 0:
            raise ValueError("Hidden layers cannot be empty.")

        if any(v <= 0 for v in values):
            raise ValueError("Hidden layer sizes must be positive.")

        return values

    except Exception:
        raise ValueError("Hidden layers must look like this: 8,4 or 16,8,4")


def split_data(X, y, test_size=0.2, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_line_chart(history, key_train, key_test, title, y_label):
    fig = go.Figure()
    x = list(range(1, len(history[key_train]) + 1))

    fig.add_trace(go.Scatter(
        x=x,
        y=history[key_train],
        mode="lines",
        name="train",
        line=dict(color=COLORS["purple"], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=history[key_test],
        mode="lines",
        name="test",
        line=dict(color=COLORS["coral"], width=2, dash="dot")
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_label,
        height=340
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

    return fig


def make_network_diagram(layer_sizes):
    fig = go.Figure()
    x_positions = np.linspace(0, 1, len(layer_sizes))

    for layer_index, layer_size in enumerate(layer_sizes):
        shown = min(layer_size, 8)
        y_positions = np.linspace(-1, 1, shown)
        x = x_positions[layer_index]

        if layer_index < len(layer_sizes) - 1:
            next_size = min(layer_sizes[layer_index + 1], 8)
            next_y_positions = np.linspace(-1, 1, next_size)
            next_x = x_positions[layer_index + 1]

            for y1 in y_positions:
                for y2 in next_y_positions:
                    fig.add_trace(go.Scatter(
                        x=[x, next_x],
                        y=[y1, y2],
                        mode="lines",
                        line=dict(color="rgba(124,106,247,0.12)", width=1),
                        showlegend=False,
                        hoverinfo="none"
                    ))

        if layer_index == 0:
            color = COLORS["teal"]
            label = "Input"
        elif layer_index == len(layer_sizes) - 1:
            color = COLORS["coral"]
            label = "Output"
        else:
            color = COLORS["purple"]
            label = f"Hidden {layer_index}"

        fig.add_trace(go.Scatter(
            x=[x] * shown,
            y=y_positions,
            mode="markers",
            marker=dict(size=19, color=color),
            showlegend=False,
            hoverinfo="none"
        ))

        more_text = f"<br>+{layer_size - 8} more" if layer_size > 8 else ""

        fig.add_annotation(
            x=x,
            y=1.38,
            text=f"{label}<br>{layer_size} neurons{more_text}",
            showarrow=False,
            font=dict(color=COLORS["text2"], size=11)
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-1.55, 1.65])
    )

    return fig


def pca_2d(X):
    X_centered = X - np.mean(X, axis=0)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    components = vt[:2]
    X_2d = X_centered @ components.T
    return X_2d, components, np.mean(X, axis=0)


def inverse_pca_2d(X_2d, components, mean):
    return X_2d @ components + mean


def make_decision_boundary(model, X, y, problem_type, feature_names):
    if X.shape[0] == 0:
        raise ValueError("Cannot draw decision boundary because X is empty.")

    if problem_type == "regression":
        raise ValueError("Decision boundary is only for classification.")

    if X.shape[1] == 1:
        raise ValueError("Decision boundary requires at least 2 features.")

    if X.shape[1] > 2:
        X_2d, components, mean = pca_2d(X)
        use_pca = True
        x_label = "PCA 1"
        y_label = "PCA 2"
        title = "Decision Boundary PCA 2D"
    else:
        X_2d = X
        use_pca = False
        x_label = feature_names[0]
        y_label = feature_names[1]
        title = "Decision Boundary 2D"

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 160),
        np.linspace(y_min, y_max, 160)
    )

    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    if use_pca:
        grid = inverse_pca_2d(grid_2d, components, mean)
    else:
        grid = grid_2d

    predictions = model.predict(grid)

    if problem_type == "multiclass_classification":
        Z = predictions.reshape(xx.shape)
        y_plot = np.argmax(y, axis=1)
    else:
        Z = predictions.reshape(-1).reshape(xx.shape)
        y_plot = y.reshape(-1)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 160),
        y=np.linspace(y_min, y_max, 160),
        z=Z,
        colorscale="Viridis",
        opacity=0.35,
        showscale=False,
        contours=dict(showlines=False)
    ))

    fig.add_trace(go.Scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        mode="markers",
        marker=dict(
            color=y_plot,
            size=8,
            line=dict(color="white", width=0.5)
        ),
        name="Data"
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=430
    )

    return fig


def make_confusion_matrix_plot(cm):
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=430
    )

    return fig


def get_confusion_matrix_array(y_true, y_pred):
    result = confusion_matrix(y_true, y_pred)

    if isinstance(result, tuple):
        cm, labels = result
        return cm, labels

    labels = np.unique(np.concatenate([np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)]))
    return result, labels


def train_perceptron_model(model, X, y, test_size, model_type):
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    if len(np.unique(y_train)) < 2:
        raise ValueError("Train split contains only one class. Reduce test size or use a larger dataset.")

    if len(np.unique(y_test)) < 1:
        raise ValueError("Test split is empty. Reduce test size or use a larger dataset.")

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train).reshape(-1)
    test_pred = model.predict(X_test).reshape(-1)

    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)

    if model_type == "Historical Perceptron":
        train_loss_history = list(model.errors_history)
        if len(train_loss_history) == 0:
            train_loss_history = [1 - train_acc]
    else:
        train_loss_history = list(model.loss_history)
        if len(train_loss_history) == 0:
            train_loss_history = [1 - train_acc]

    history = {
        "train_loss": train_loss_history,
        "test_loss": [1 - test_acc for _ in train_loss_history],
        "train_metric": [train_acc for _ in train_loss_history],
        "test_metric": [test_acc for _ in train_loss_history],
        "train_accuracy": [train_acc for _ in train_loss_history],
        "test_accuracy": [test_acc for _ in train_loss_history],
        "test_precision": [precision(y_test, test_pred, average="binary") for _ in train_loss_history],
        "test_recall": [recall(y_test, test_pred, average="binary") for _ in train_loss_history],
        "test_f1": [f1_score(y_test, test_pred, average="binary") for _ in train_loss_history]
    }

    return history, X_train, X_test, y_train, y_test, train_pred, test_pred


def train_mlp_for_experiment(
    X,
    y,
    config,
    problem_type,
    hidden_layers,
    activation,
    learning_rate,
    epochs,
    test_size,
    l2_lambda=0.0,
    dropout_rate=0.0,
    early_stopping=False,
    patience=20
):
    model = MLP(
        input_size=config["input_size"],
        hidden_layers=hidden_layers,
        output_size=config["output_size"],
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation,
        output_activation=config["output_activation"],
        loss=config["loss"],
        l2_lambda=l2_lambda,
        dropout_rate=dropout_rate
    )

    trainer = Trainer(model=model, problem_type=problem_type)

    history = trainer.train(
        X,
        y,
        test_size=test_size,
        early_stopping=early_stopping,
        patience=patience
    )

    y_pred = model.predict(trainer.X_test)

    return {
        "model": model,
        "trainer": trainer,
        "history": history,
        "X_train": trainer.X_train,
        "X_test": trainer.X_test,
        "y_train": trainer.y_train,
        "y_test": trainer.y_test,
        "y_pred": y_pred,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_train_metric": history["train_metric"][-1],
        "final_test_metric": history["test_metric"][-1],
    }


def metric_name(problem_type):
    if problem_type == "regression":
        return "MSE"
    return "Accuracy"


# ============================================================
# Session state
# ============================================================

default_state = {
    "trained": False,
    "model": None,
    "model_type": None,
    "history": None,
    "X": None,
    "y": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "y_pred": None,
    "config": None,
    "problem_type": None,
    "hidden_layers": None,
    "error_message": None
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:22px">
        <div style="font-size:30px">🧠</div>
        <div>
            <div style="font-size:21px;font-weight:800">NeuroLearn</div>
            <div style="font-size:11px;color:#8890a4">educational neural network lab</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown('<div class="section-title">Problem type</div>', unsafe_allow_html=True)
    problem_type = st.selectbox(
        "Problem type",
        [
            "binary_classification",
            "multiclass_classification",
            "regression"
        ],
        format_func=lambda x: {
            "binary_classification": "Binary classification",
            "multiclass_classification": "Multi-class classification",
            "regression": "Regression"
        }[x],
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-title">Mode</div>', unsafe_allow_html=True)
    app_mode = st.radio(
        "Mode",
        [
            "Manual Training",
            "Experiment 1: Perceptron vs MLP",
            "Experiment 2: Effect of Layers",
            "Experiment 3: Regularization"
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)

    if app_mode == "Manual Training":
        model_type = st.radio(
            "Model",
            [
                "MLP",
                "Historical Perceptron",
                "Activated Perceptron"
            ],
            label_visibility="collapsed"
        )
    else:
        model_type = "MLP"
        st.markdown(
            '<div class="info-box">Experiment mode uses predefined model comparisons.</div>',
            unsafe_allow_html=True
        )

    if model_type != "MLP" and problem_type != "binary_classification":
        st.markdown(
            '<div class="warn-box">Perceptron models are only available for binary classification.</div>',
            unsafe_allow_html=True
        )

    if app_mode == "Manual Training":
        if model_type == "MLP":
            st.markdown('<div class="section-title">Architecture</div>', unsafe_allow_html=True)
            hidden_layers_text = st.text_input("Hidden layers", value="8,4")

            st.markdown('<div class="section-title">Activation</div>', unsafe_allow_html=True)
            activation = st.selectbox(
                "Hidden activation",
                ["relu", "sigmoid", "tanh"],
                format_func=lambda x: {
                    "relu": "ReLU",
                    "sigmoid": "Sigmoid",
                    "tanh": "Tanh"
                }[x]
            )
        else:
            hidden_layers_text = ""
            activation = "sigmoid"

            if model_type == "Activated Perceptron":
                st.markdown('<div class="section-title">Activation</div>', unsafe_allow_html=True)
                activation = st.selectbox(
                    "Activation",
                    ["sigmoid", "tanh", "relu"],
                    format_func=lambda x: {
                        "sigmoid": "Sigmoid",
                        "tanh": "Tanh",
                        "relu": "ReLU"
                    }[x]
                )
    else:
        hidden_layers_text = ""
        activation = "relu"

    st.markdown('<div class="section-title">Training</div>', unsafe_allow_html=True)

    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        value=0.01
    )

    epochs = st.slider("Epochs", 10, 20000, 1000, step=10)

    test_size = st.slider(
        "Test size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05
    )

    if app_mode == "Experiment 1: Perceptron vs MLP":
        st.markdown('<div class="section-title">Experiment 1 Parameters</div>', unsafe_allow_html=True)
        exp1_hidden_layers_text = st.text_input("MLP architecture", value="8,4")
        exp1_activation = st.selectbox("MLP activation", ["relu", "sigmoid", "tanh"])

    elif app_mode == "Experiment 2: Effect of Layers":
        st.markdown('<div class="section-title">Experiment 2 Architectures</div>', unsafe_allow_html=True)
        arch1_text = st.text_input("Architecture 1", value="8")
        arch2_text = st.text_input("Architecture 2", value="16,8")
        arch3_text = st.text_input("Architecture 3", value="32,16,8")
        exp2_activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"])

    elif app_mode == "Experiment 3: Regularization":
        st.markdown('<div class="section-title">Experiment 3 Parameters</div>', unsafe_allow_html=True)
        exp3_hidden_layers_text = st.text_input("Architecture", value="32,16")
        exp3_activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"])

    st.markdown('<div class="section-title">Regularization</div>', unsafe_allow_html=True)

    # Default values used by both Manual Training and Experiment 3
    use_l2 = False
    l2_lambda = 0.0
    use_dropout = False
    dropout_rate = 0.0
    use_early_stopping = False
    early_stopping_patience = 20

    if app_mode == "Manual Training" and model_type == "MLP":
        use_l2 = st.checkbox("Use L2 Regularization", value=False)
        if use_l2:
            l2_lambda = st.slider(
                "L2 Lambda",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.001
            )

        use_dropout = st.checkbox("Use Dropout", value=False)
        if use_dropout:
            dropout_rate = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.8,
                value=0.2,
                step=0.05
            )

        use_early_stopping = st.checkbox("Use Early Stopping", value=False)
        if use_early_stopping:
            early_stopping_patience = st.slider(
                "Early Stopping Patience",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )

    elif app_mode == "Experiment 3: Regularization":
        st.info("This experiment compares: no regularization, L2, Dropout, and Early Stopping.")
        l2_lambda = st.slider("L2 lambda for comparison", 0.0, 0.1, 0.01, step=0.005)
        dropout_rate = st.slider("Dropout rate for comparison", 0.0, 0.8, 0.2, step=0.05)
        early_stopping_patience = st.slider("Early stopping patience", 5, 100, 20, step=5)

    else:
        st.info("Regularization is available for MLP manual training and Experiment 3.")

    if app_mode == "Manual Training":
        train_button = st.button("▶ Train model", type="primary", use_container_width=True)
    else:
        train_button = st.button("▶ Run experiment", type="primary", use_container_width=True)


# ============================================================
# Header
# ============================================================

col_status, col_title, col_backend = st.columns([1, 2, 1])

with col_status:
    if st.session_state.trained:
        st.markdown('<div class="status-pill"><span class="dot-green"></span>trained</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill"><span class="dot-gray"></span>ready</div>', unsafe_allow_html=True)

with col_title:
    st.markdown(
        "<h2 style='text-align:center;margin:0'>Neural Network Educational Platform</h2>",
        unsafe_allow_html=True
    )

with col_backend:
    st.markdown(
        "<div style='text-align:right;color:#8890a4;padding-top:7px'>NumPy backend</div>",
        unsafe_allow_html=True
    )

st.divider()


# ============================================================
# Dataset + training
# ============================================================

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center;padding:75px;color:#8890a4">
        <div style="font-size:58px">🧠</div>
        <h3>Configure your model in the sidebar</h3>
        <p>Upload a CSV, choose the target column, select a model, then train.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-box">
        <b>Models</b><br>
        Historical Perceptron<br>
        Activated Perceptron<br>
        MLP
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-box">
        <b>Learning</b><br>
        Forward propagation<br>
        Backpropagation<br>
        Gradient descent
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-box">
        <b>Visualizations</b><br>
        Loss curves<br>
        Accuracy curves<br>
        Decision boundary
        </div>
        """, unsafe_allow_html=True)

else:
    try:
        df = pd.read_csv(uploaded_file)

        st.markdown(
            f'<div class="info-box">✓ Dataset loaded: {uploaded_file.name} · {df.shape[0]} rows · {df.shape[1]} columns</div>',
            unsafe_allow_html=True
        )

        target_column = st.selectbox("Target column", df.columns)

        with st.expander("Dataset preview", expanded=False):
            st.dataframe(df.head())

        if train_button:
            try:
                if model_type != "MLP" and problem_type != "binary_classification":
                    st.error("Perceptron models only support binary classification.")
                    st.stop()

                X, y, config = prepare_data(
                    df=df,
                    target_column=target_column,
                    problem_type=problem_type
                )

                if X.shape[0] < 4:
                    raise ValueError("Dataset is too small. Please use at least 4 rows.")

                # ============================================================
                # Experiment 1: Perceptron vs MLP
                # ============================================================

                if app_mode == "Experiment 1: Perceptron vs MLP":
                    if problem_type != "binary_classification":
                        st.error("Experiment 1 uses Perceptron, so it only works with binary classification.")
                        st.stop()

                    hidden_layers = safe_parse_hidden_layers(exp1_hidden_layers_text)

                    st.subheader("Experiment 1: Perceptron vs MLP")

                    with st.spinner("Training Perceptron and MLP..."):
                        perceptron_model = Perceptron(
                            learning_rate=learning_rate,
                            epochs=epochs
                        )

                        history_p, X_train_p, X_test_p, y_train_p, y_test_p, train_pred_p, test_pred_p = train_perceptron_model(
                            model=perceptron_model,
                            X=X,
                            y=y,
                            test_size=test_size,
                            model_type="Historical Perceptron"
                        )

                        mlp_result = train_mlp_for_experiment(
                            X=X,
                            y=y,
                            config=config,
                            problem_type=problem_type,
                            hidden_layers=hidden_layers,
                            activation=exp1_activation,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            test_size=test_size
                        )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Perceptron Test Accuracy", f"{history_p['test_metric'][-1] * 100:.2f}%")
                    c2.metric("MLP Test Accuracy", f"{mlp_result['final_test_metric'] * 100:.2f}%")
                    c3.metric("Perceptron Final Loss", f"{history_p['train_loss'][-1]:.4f}")
                    c4.metric("MLP Final Test Loss", f"{mlp_result['final_test_loss']:.4f}")

                    comparison_df = pd.DataFrame([
                        {
                            "Model": "Historical Perceptron",
                            "Architecture": "Single linear unit",
                            "Train metric": history_p["train_metric"][-1],
                            "Test metric": history_p["test_metric"][-1],
                            "Final train loss": history_p["train_loss"][-1],
                            "Final test loss": history_p["test_loss"][-1]
                        },
                        {
                            "Model": "MLP",
                            "Architecture": str(hidden_layers),
                            "Train metric": mlp_result["final_train_metric"],
                            "Test metric": mlp_result["final_test_metric"],
                            "Final train loss": mlp_result["final_train_loss"],
                            "Final test loss": mlp_result["final_test_loss"]
                        }
                    ])

                    st.dataframe(comparison_df, use_container_width=True)

                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        y=history_p["train_loss"],
                        name="Perceptron loss",
                        line=dict(color=COLORS["amber"])
                    ))
                    fig_loss.add_trace(go.Scatter(
                        y=mlp_result["history"]["train_loss"],
                        name="MLP train loss",
                        line=dict(color=COLORS["purple"])
                    ))
                    fig_loss.add_trace(go.Scatter(
                        y=mlp_result["history"]["test_loss"],
                        name="MLP test loss",
                        line=dict(color=COLORS["coral"], dash="dot")
                    ))
                    fig_loss.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Experiment 1 - Loss Comparison",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=420
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

                    if mlp_result["final_test_metric"] > history_p["test_metric"][-1]:
                        st.markdown("""
                        <div class="info-box">
                        The MLP performs better than the Perceptron. This usually means that the dataset contains
                        non-linear patterns that cannot be learned well by a simple linear model.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warn-box">
                        The Perceptron performs similarly or better. This may mean that the dataset is simple or mostly linearly separable.
                        </div>
                        """, unsafe_allow_html=True)

                    st.stop()

                # ============================================================
                # Experiment 2: Effect of Layers
                # ============================================================

                if app_mode == "Experiment 2: Effect of Layers":
                    architectures = {
                        "Architecture 1": safe_parse_hidden_layers(arch1_text),
                        "Architecture 2": safe_parse_hidden_layers(arch2_text),
                        "Architecture 3": safe_parse_hidden_layers(arch3_text)
                    }

                    st.subheader("Experiment 2: Effect of Number of Layers")

                    results = []
                    fig_loss = go.Figure()
                    fig_metric = go.Figure()

                    with st.spinner("Training different MLP architectures..."):
                        for name, layers in architectures.items():
                            result = train_mlp_for_experiment(
                                X=X,
                                y=y,
                                config=config,
                                problem_type=problem_type,
                                hidden_layers=layers,
                                activation=exp2_activation,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                test_size=test_size
                            )

                            results.append({
                                "Model": name,
                                "Architecture": str(layers),
                                "Number of hidden layers": len(layers),
                                "Train metric": result["final_train_metric"],
                                "Test metric": result["final_test_metric"],
                                "Train loss": result["final_train_loss"],
                                "Test loss": result["final_test_loss"],
                                "Generalization gap": abs(result["final_train_metric"] - result["final_test_metric"])
                            })

                            fig_loss.add_trace(go.Scatter(
                                y=result["history"]["train_loss"],
                                name=f"{name} train loss"
                            ))

                            fig_loss.add_trace(go.Scatter(
                                y=result["history"]["test_loss"],
                                name=f"{name} test loss",
                                line=dict(dash="dot")
                            ))

                            fig_metric.add_trace(go.Scatter(
                                y=result["history"]["train_metric"],
                                name=f"{name} train {metric_name(problem_type)}"
                            ))

                            fig_metric.add_trace(go.Scatter(
                                y=result["history"]["test_metric"],
                                name=f"{name} test {metric_name(problem_type)}",
                                line=dict(dash="dot")
                            ))

                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    best_row = results_df.sort_values(by="Test metric", ascending=False).iloc[0]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best architecture", best_row["Architecture"])
                    c2.metric(f"Best test {metric_name(problem_type)}", f"{best_row['Test metric']:.4f}")
                    c3.metric("Generalization gap", f"{best_row['Generalization gap']:.4f}")

                    fig_loss.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Experiment 2 - Loss Comparison by Architecture",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=420
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

                    fig_metric.update_layout(
                        **PLOTLY_LAYOUT,
                        title=f"Experiment 2 - {metric_name(problem_type)} Comparison",
                        xaxis_title="Epoch",
                        yaxis_title=metric_name(problem_type),
                        height=420
                    )
                    st.plotly_chart(fig_metric, use_container_width=True)

                    st.markdown("""
                    <div class="info-box">
                    This experiment shows how the number of hidden layers changes the learning capacity of the model.
                    A deeper model can learn more complex patterns, but if the gap between train and test performance becomes large,
                    the model may be overfitting.
                    </div>
                    """, unsafe_allow_html=True)

                    st.stop()

                # ============================================================
                # Experiment 3: Regularization
                # ============================================================

                if app_mode == "Experiment 3: Regularization":
                    hidden_layers = safe_parse_hidden_layers(exp3_hidden_layers_text)

                    configs_exp = [
                        {
                            "name": "Without regularization",
                            "l2_lambda": 0.0,
                            "dropout_rate": 0.0,
                            "early_stopping": False
                        },
                        {
                            "name": "With L2 regularization",
                            "l2_lambda": l2_lambda,
                            "dropout_rate": 0.0,
                            "early_stopping": False
                        },
                        {
                            "name": "With Dropout",
                            "l2_lambda": 0.0,
                            "dropout_rate": dropout_rate,
                            "early_stopping": False
                        },
                        {
                            "name": "With Early Stopping",
                            "l2_lambda": 0.0,
                            "dropout_rate": 0.0,
                            "early_stopping": True
                        }
                    ]

                    st.subheader("Experiment 3: Overfitting vs Regularization")

                    results = []
                    fig_loss = go.Figure()
                    fig_metric = go.Figure()

                    with st.spinner("Training regularized and non-regularized models..."):
                        for exp_config in configs_exp:
                            name = exp_config["name"]

                            result = train_mlp_for_experiment(
                                X=X,
                                y=y,
                                config=config,
                                problem_type=problem_type,
                                hidden_layers=hidden_layers,
                                activation=exp3_activation,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                test_size=test_size,
                                l2_lambda=exp_config["l2_lambda"],
                                dropout_rate=exp_config["dropout_rate"],
                                early_stopping=exp_config["early_stopping"],
                                patience=early_stopping_patience
                            )

                            gap = abs(result["final_train_metric"] - result["final_test_metric"])

                            results.append({
                                "Model": name,
                                "Architecture": str(hidden_layers),
                                "L2 lambda": exp_config["l2_lambda"],
                                "Dropout rate": exp_config["dropout_rate"],
                                "Early stopping": exp_config["early_stopping"],
                                "Train metric": result["final_train_metric"],
                                "Test metric": result["final_test_metric"],
                                "Train loss": result["final_train_loss"],
                                "Test loss": result["final_test_loss"],
                                "Generalization gap": gap
                            })

                            fig_loss.add_trace(go.Scatter(
                                y=result["history"]["train_loss"],
                                name=f"{name} train loss"
                            ))

                            fig_loss.add_trace(go.Scatter(
                                y=result["history"]["test_loss"],
                                name=f"{name} test loss",
                                line=dict(dash="dot")
                            ))

                            fig_metric.add_trace(go.Scatter(
                                y=result["history"]["train_metric"],
                                name=f"{name} train {metric_name(problem_type)}"
                            ))

                            fig_metric.add_trace(go.Scatter(
                                y=result["history"]["test_metric"],
                                name=f"{name} test {metric_name(problem_type)}",
                                line=dict(dash="dot")
                            ))

                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    best_gap_row = results_df.sort_values(by="Generalization gap", ascending=True).iloc[0]
                    best_metric_row = results_df.sort_values(by="Test metric", ascending=False).iloc[0]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best test metric", best_metric_row["Model"])
                    c2.metric("Best generalization gap", best_gap_row["Model"])
                    c3.metric("L2 / Dropout", f"{l2_lambda:.3f} / {dropout_rate:.2f}")

                    fig_loss.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Experiment 3 - Train vs Test Loss",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=420
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

                    fig_metric.update_layout(
                        **PLOTLY_LAYOUT,
                        title=f"Experiment 3 - Train vs Test {metric_name(problem_type)}",
                        xaxis_title="Epoch",
                        yaxis_title=metric_name(problem_type),
                        height=420
                    )
                    st.plotly_chart(fig_metric, use_container_width=True)

                    st.markdown(f"""
                    <div class="info-box">
                    This experiment compares four strategies: no regularization, L2, Dropout, and Early Stopping.
                    The best test metric was achieved by <b>{best_metric_row['Model']}</b>.
                    The smallest generalization gap was achieved by <b>{best_gap_row['Model']}</b>.
                    </div>
                    """, unsafe_allow_html=True)

                    st.stop()

                # ============================================================
                # Manual Training
                # ============================================================

                if model_type == "MLP":
                    hidden_layers = safe_parse_hidden_layers(hidden_layers_text)

                    model = MLP(
                        input_size=config["input_size"],
                        hidden_layers=hidden_layers,
                        output_size=config["output_size"],
                        learning_rate=learning_rate,
                        epochs=epochs,
                        activation=activation,
                        output_activation=config["output_activation"],
                        loss=config["loss"],
                        l2_lambda=l2_lambda,
                        dropout_rate=dropout_rate
                    )

                    trainer = Trainer(model=model, problem_type=problem_type)

                    progress_bar = st.progress(0)
                    status_box = st.empty()
                    live_chart_box = st.empty()

                    def live_training_update(epoch, history):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(min(progress, 1.0))

                        if problem_type in ["binary_classification", "multiclass_classification"]:
                            status_box.markdown(
                                f"""
                                <div class="info-box">
                                Epoch {epoch + 1}/{epochs} |
                                Train loss: {history["train_loss"][-1]:.4f} |
                                Test loss: {history["test_loss"][-1]:.4f} |
                                Train acc: {history["train_accuracy"][-1]:.4f} |
                                Test acc: {history["test_accuracy"][-1]:.4f} |
                                F1: {history["test_f1"][-1]:.4f}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            status_box.markdown(
                                f"""
                                <div class="info-box">
                                Epoch {epoch + 1}/{epochs} |
                                Train loss: {history["train_loss"][-1]:.4f} |
                                Test loss: {history["test_loss"][-1]:.4f} |
                                Test MSE: {history["test_metric"][-1]:.4f}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                y=history["train_loss"],
                                name="Train loss",
                                line=dict(color=COLORS["purple"])
                            ))

                            fig.add_trace(go.Scatter(
                                y=history["test_loss"],
                                name="Test loss",
                                line=dict(color=COLORS["coral"], dash="dot")
                            ))

                            fig.update_layout(
                                **PLOTLY_LAYOUT,
                                title="Live Training Loss",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=300
                            )

                            live_chart_box.plotly_chart(fig, use_container_width=True)

                    with st.spinner("Training MLP..."):
                        history = trainer.train(
                            X,
                            y,
                            test_size=test_size,
                            callback=live_training_update,
                            early_stopping=use_early_stopping,
                            patience=early_stopping_patience
                        )

                    y_pred = model.predict(trainer.X_test)

                    st.session_state.X_train = trainer.X_train
                    st.session_state.X_test = trainer.X_test
                    st.session_state.y_train = trainer.y_train
                    st.session_state.y_test = trainer.y_test
                    st.session_state.y_pred = y_pred

                elif model_type == "Historical Perceptron":
                    hidden_layers = []

                    model = Perceptron(
                        learning_rate=learning_rate,
                        epochs=epochs
                    )

                    with st.spinner("Training Historical Perceptron..."):
                        history, X_train, X_test, y_train, y_test, train_pred, test_pred = train_perceptron_model(
                            model=model,
                            X=X,
                            y=y,
                            test_size=test_size,
                            model_type=model_type
                        )

                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = test_pred

                elif model_type == "Activated Perceptron":
                    hidden_layers = []

                    model = PerceptronActivation(
                        learning_rate=learning_rate,
                        epochs=epochs,
                        activation=activation,
                        loss="binary_cross_entropy"
                    )

                    with st.spinner("Training Activated Perceptron..."):
                        history, X_train, X_test, y_train, y_test, train_pred, test_pred = train_perceptron_model(
                            model=model,
                            X=X,
                            y=y,
                            test_size=test_size,
                            model_type=model_type
                        )

                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = test_pred

                else:
                    raise ValueError("Unknown model type.")

                st.session_state.trained = True
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.history = history
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.config = config
                st.session_state.problem_type = problem_type
                st.session_state.hidden_layers = hidden_layers
                st.session_state.error_message = None

                st.rerun()

            except ValueError as e:
                st.session_state.error_message = str(e)
                st.error(str(e))

            except Exception as e:
                st.session_state.error_message = str(e)
                st.error(f"Unexpected error: {e}")

    except Exception as e:
        st.error(f"Could not read CSV file: {e}")


# ============================================================
# Results
# ============================================================

if st.session_state.trained:
    model = st.session_state.model
    model_type = st.session_state.model_type
    history = st.session_state.history
    X = st.session_state.X
    y = st.session_state.y
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    config = st.session_state.config
    problem_type = st.session_state.problem_type
    hidden_layers = st.session_state.hidden_layers

    st.markdown(
        f'<div class="info-box">Model: <b>{model_type}</b> · Problem: <b>{problem_type}</b> · Loss: <b>{config["loss"]}</b> · Output: <b>{config["output_activation"]}</b></div>',
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "📈 Training curves",
        "🗺 Decision boundary",
        "🔮 Network view",
        "📊 Metrics",
        "🔢 Confusion matrix"
    ])

    with tabs[0]:
        c1, c2 = st.columns(2)

        with c1:
            fig_loss = make_line_chart(
                history,
                "train_loss",
                "test_loss",
                "Loss vs Epochs",
                "Loss"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with c2:
            metric_label = "Accuracy" if problem_type != "regression" else "MSE"
            fig_metric = make_line_chart(
                history,
                "train_metric",
                "test_metric",
                f"{metric_label} vs Epochs",
                metric_label
            )
            st.plotly_chart(fig_metric, use_container_width=True)

        if problem_type in ["binary_classification", "multiclass_classification"]:
            st.subheader("Classification Metrics Evolution")

            fig_cls = go.Figure()

            fig_cls.add_trace(go.Scatter(
                y=history["test_accuracy"],
                name="Test Accuracy",
                line=dict(color=COLORS["teal"], width=2)
            ))

            fig_cls.add_trace(go.Scatter(
                y=history["test_precision"],
                name="Test Precision",
                line=dict(color=COLORS["purple"], width=2)
            ))

            fig_cls.add_trace(go.Scatter(
                y=history["test_recall"],
                name="Test Recall",
                line=dict(color=COLORS["amber"], width=2)
            ))

            fig_cls.add_trace(go.Scatter(
                y=history["test_f1"],
                name="Test F1-score",
                line=dict(color=COLORS["coral"], width=2)
            ))

            fig_cls.update_layout(
                **PLOTLY_LAYOUT,
                title="Classification Metrics vs Epochs",
                xaxis_title="Epoch",
                yaxis_title="Score",
                height=360
            )

            fig_cls.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig_cls.update_yaxes(gridcolor="rgba(255,255,255,0.06)", range=[0, 1.05])

            st.plotly_chart(fig_cls, use_container_width=True)

        train_loss = history["train_loss"][-1]
        test_loss = history["test_loss"][-1]

        if problem_type != "regression" and test_loss > train_loss * 1.5:
            st.markdown(
                '<div class="warn-box">⚠ Possible overfitting: test loss is much higher than train loss.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="info-box">✓ Training completed successfully.</div>',
                unsafe_allow_html=True
            )

    with tabs[1]:
        try:
            if problem_type == "regression":
                st.info("Decision boundary is only available for classification.")
            else:
                fig_boundary = make_decision_boundary(
                    model=model,
                    X=X,
                    y=y,
                    problem_type=problem_type,
                    feature_names=config["feature_names"]
                )
                st.plotly_chart(fig_boundary, use_container_width=True)

        except Exception as e:
            st.warning(f"Decision boundary unavailable: {e}")

    with tabs[2]:
        if model_type == "MLP":
            layer_sizes = [config["input_size"]] + hidden_layers + [config["output_size"]]
            fig_net = make_network_diagram(layer_sizes)
            st.plotly_chart(fig_net, use_container_width=True)

            total_params = 0
            for key, value in model.parameters.items():
                total_params += value.size

            c1, c2, c3 = st.columns(3)
            c1.metric("Total parameters", f"{total_params:,}")
            c2.metric("Hidden layers", len(hidden_layers))
            c3.metric("Activation", model.activation_name)
        else:
            st.info("Network view is mainly useful for MLP. Perceptron has one linear decision unit.")

    with tabs[3]:
        if problem_type == "binary_classification":
            y_true = y_test.reshape(-1).astype(int)
            y_pred_class = y_pred.reshape(-1).astype(int)

            acc = accuracy(y_true, y_pred_class)
            prec = precision(y_true, y_pred_class, average="binary")
            rec = recall(y_true, y_pred_class, average="binary")
            f1 = f1_score(y_true, y_pred_class, average="binary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc * 100:.1f}%")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1-score", f"{f1:.3f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Final train loss", f"{history['train_loss'][-1]:.4f}")
            c6.metric("Final test loss", f"{history['test_loss'][-1]:.4f}")
            c7.metric("Train accuracy", f"{history['train_accuracy'][-1] * 100:.1f}%")
            c8.metric("Test accuracy", f"{history['test_accuracy'][-1] * 100:.1f}%")

        elif problem_type == "multiclass_classification":
            y_true = np.argmax(y_test, axis=1)
            y_pred_class = y_pred.reshape(-1).astype(int)

            acc = accuracy(y_true, y_pred_class)
            prec = precision(y_true, y_pred_class, average="macro")
            rec = recall(y_true, y_pred_class, average="macro")
            f1 = f1_score(y_true, y_pred_class, average="macro")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc * 100:.1f}%")
            c2.metric("Precision macro", f"{prec:.3f}")
            c3.metric("Recall macro", f"{rec:.3f}")
            c4.metric("F1-score macro", f"{f1:.3f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Final train loss", f"{history['train_loss'][-1]:.4f}")
            c6.metric("Final test loss", f"{history['test_loss'][-1]:.4f}")
            c7.metric("Classes", config["output_size"])
            c8.metric("Loss", config["loss"])

        elif problem_type == "regression":
            y_true = y_test.reshape(-1)
            y_pred_reg = y_pred.reshape(-1)

            c1, c2, c3 = st.columns(3)
            c1.metric("MSE", f"{mse(y_true, y_pred_reg):.4f}")
            c2.metric("RMSE", f"{rmse(y_true, y_pred_reg):.4f}")
            c3.metric("R² score", f"{r2_score(y_true, y_pred_reg):.3f}")

    with tabs[4]:
        if problem_type == "regression":
            st.info("Confusion matrix is not applicable for regression.")
        else:
            if problem_type == "binary_classification":
                y_true = y_test.reshape(-1).astype(int)
                y_pred_class = y_pred.reshape(-1).astype(int)
            else:
                y_true = np.argmax(y_test, axis=1)
                y_pred_class = y_pred.reshape(-1).astype(int)

            cm, labels = get_confusion_matrix_array(y_true, y_pred_class)

            fig_cm = make_confusion_matrix_plot(cm)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.write("Labels:", labels)