"""
EEG-Based Cognitive Load Detection – Streamlit Dashboard
=========================================================
Interactive web app for exploring EEG stress-level classification results
from both Machine Learning and Deep Learning models.
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"
DATA_PATH = ROOT / "processed_dataset.csv"
DL_RESULTS_PATH = ROOT / "dl_results.json"
ML_RESULTS_PATH = ROOT / "ml_results.json"

STRESS_NAMES = ["Natural", "Low-Level", "Mid-Level", "High-Level"]
STRESS_MAP = {0: "Natural", 1: "Low-Level", 2: "Mid-Level", 3: "High-Level"}
STRESS_COLORS = {"Natural": "#2ecc71", "Low-Level": "#f1c40f", "Mid-Level": "#e67e22", "High-Level": "#e74c3c"}


# ── DL Model Definitions (must match training) ────────────────────────────
if TORCH_AVAILABLE:
    class DNN(nn.Module):
        def __init__(self, n_features, n_classes=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    class CNN1D(nn.Module):
        def __init__(self, n_features, n_classes=4):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x).squeeze(-1)
            return self.fc(x)

    class LSTMModel(nn.Module):
        def __init__(self, n_features, n_classes=4):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,
                                batch_first=True, dropout=0.3, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            x = x.unsqueeze(-1)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


# ── Helper loaders (cached) ───────────────────────────────────────────────
@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_json_results(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_scaler_and_cols():
    scaler_path = MODEL_DIR / "scaler.pkl"
    cols_path = MODEL_DIR / "feature_cols.pkl"
    if scaler_path.exists() and cols_path.exists():
        return joblib.load(scaler_path), joblib.load(cols_path)
    return None, None


@st.cache_resource
def load_ml_model(name):
    path = MODEL_DIR / f"ml_{name}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def load_dl_model(name, n_features):
    if not TORCH_AVAILABLE:
        return None
    path = MODEL_DIR / f"dl_{name}.pt"
    if not path.exists():
        return None
    model_map = {"DNN": DNN, "CNN_1D": CNN1D, "LSTM": LSTMModel}
    cls = model_map.get(name)
    if cls is None:
        return None
    model = cls(n_features)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Cognitive Load Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Navigation ────────────────────────────────────────────────────
st.sidebar.title("🧠 EEG Stress Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Data Explorer", "🤖 ML Results", "🧬 DL Results", "🔮 Predict"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.info("EEG-Based Cognitive Load Detection using ML & DL models.")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🧠 EEG-Based Cognitive Load Detection")
    st.markdown(
        """
        This dashboard presents an end-to-end pipeline for **detecting cognitive stress levels**
        from EEG brain signals collected during **Arithmetic** and **Stroop** tasks.

        ---
        ### 📋 Project Summary
        | Aspect | Detail |
        |--------|--------|
        | **Data Source** | Muse EEG headband (4 EEG + 2 AUX + 3 ACC channels) |
        | **Tasks** | Arithmetic reasoning · Stroop color-word test |
        | **Participants** | 15 per stress level per task |
        | **Stress Levels** | Natural · Low · Mid · High |
        | **Features** | 77 statistical features (mean, std, min, max, median, skew, kurtosis) |
        | **ML Models** | Random Forest · SVM · KNN · Gradient Boosting · Logistic Regression |
        | **DL Models** | DNN · 1D-CNN · LSTM |

        ---
        ### 🔬 Pipeline
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Step 1", "Raw EEG", "60 .txt files per task")
    col2.metric("Step 2", "Features", "77 extracted")
    col3.metric("Step 3", "Training", "8 models")
    col4.metric("Step 4", "Prediction", "4 classes")

    st.markdown("---")

    # Show quick dataset stats if available
    df = load_dataset()
    if df is not None:
        st.subheader("📈 Quick Dataset Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", df.shape[0])
        c2.metric("Features", df.shape[1] - 4)
        c3.metric("Tasks", df["task"].nunique())
        c4.metric("Stress Levels", df["stress_level"].nunique())


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    df = load_dataset()

    if df is None:
        st.error("Processed dataset not found. Run `data_preprocessing.py` first.")
        st.stop()

    # ── Dataset overview ───────────────────────────────────────────────
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Stress distribution ────────────────────────────────────────────
    with col1:
        st.subheader("Stress Level Distribution")
        stress_counts = df["stress_label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [STRESS_COLORS.get(n, "#3498db") for n in
                  ["Natural", "Low-Level", "Mid-Level", "High-Level"]]
        # Map label names
        label_map = {"natural": "Natural", "lowlevel": "Low-Level",
                     "midlevel": "Mid-Level", "highlevel": "High-Level"}
        mapped_counts = stress_counts.rename(index=label_map)
        bar_colors = [STRESS_COLORS.get(n, "#3498db") for n in mapped_counts.index]
        mapped_counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_xlabel("Stress Level")
        ax.set_title("Samples per Stress Level")
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Task distribution ──────────────────────────────────────────────
    with col2:
        st.subheader("Task Distribution")
        task_counts = df["task"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        task_counts.plot(kind="bar", ax=ax, color=["#3498db", "#9b59b6"], edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_xlabel("Task Type")
        ax.set_title("Samples per Task")
        plt.xticks(rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Feature distributions ──────────────────────────────────────────
    st.subheader("Feature Distribution Explorer")
    exclude = {"task", "stress_level", "stress_label", "participant"}
    feature_cols = [c for c in df.columns if c not in exclude]

    selected_feat = st.selectbox("Select a feature to visualize:", feature_cols)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    for level in sorted(df["stress_level"].unique()):
        subset = df[df["stress_level"] == level][selected_feat]
        axes[0].hist(subset, bins=20, alpha=0.6, label=STRESS_NAMES[level])
    axes[0].set_title(f"Distribution of {selected_feat}")
    axes[0].set_xlabel(selected_feat)
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Boxplot by stress level
    df_plot = df[[selected_feat, "stress_level"]].copy()
    df_plot["Stress"] = df_plot["stress_level"].map(STRESS_MAP)
    sns.boxplot(data=df_plot, x="Stress", y=selected_feat, ax=axes[1],
                palette=list(STRESS_COLORS.values()))
    axes[1].set_title(f"{selected_feat} by Stress Level")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ── Correlation heatmap (top features) ─────────────────────────────
    st.subheader("Feature Correlation Heatmap (Top 15 by variance)")
    top_feats = df[feature_cols].var().nlargest(15).index.tolist()
    corr = df[top_feats].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix – Top 15 Features by Variance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Descriptive statistics ─────────────────────────────────────────
    st.subheader("Descriptive Statistics")
    st.dataframe(df[feature_cols].describe().T, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: ML RESULTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Results":
    st.title("🤖 Machine Learning Results")

    ml_results = load_json_results(str(ML_RESULTS_PATH))

    if ml_results is None:
        st.warning(
            "ML results not found (`ml_results.json`). "
            "Please run the `train_ml.ipynb` notebook first to generate results."
        )
        st.info("Once trained, this page will show: comparison charts, confusion matrices, and per-model metrics.")
        st.stop()

    # ── Results table ──────────────────────────────────────────────────
    st.subheader("📊 Model Performance Comparison")
    res_df = pd.DataFrame(ml_results).T
    res_df.index.name = "Model"

    # Highlight best values
    st.dataframe(
        res_df.style.highlight_max(axis=0, color="#2ecc71", subset=["accuracy", "precision", "recall", "f1_score"]),
        use_container_width=True,
    )

    # ── Bar chart ──────────────────────────────────────────────────────
    st.subheader("📈 Metrics Comparison")
    metrics_to_show = st.multiselect(
        "Select metrics to compare:",
        ["accuracy", "precision", "recall", "f1_score", "cv_mean"],
        default=["accuracy", "f1_score"],
    )

    if metrics_to_show:
        fig, ax = plt.subplots(figsize=(10, 5))
        res_df[metrics_to_show].plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white")
        ax.set_ylabel("Score")
        ax.set_title("ML Model Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Confusion matrices ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📉 Confusion Matrices")

    ml_model_names = list(ml_results.keys())
    cm_cols = st.columns(min(3, len(ml_model_names)))

    for idx, name in enumerate(ml_model_names):
        cm_path = STATIC_DIR / f"cm_{name}.png"
        col = cm_cols[idx % 3]
        with col:
            if cm_path.exists():
                st.image(str(cm_path), caption=f"{name}", use_container_width=True)
            else:
                st.caption(f"No confusion matrix image for {name}")

    # ── ML comparison chart (if saved) ─────────────────────────────────
    ml_comp_path = STATIC_DIR / "ml_comparison.png"
    if ml_comp_path.exists():
        st.markdown("---")
        st.subheader("Overall Comparison Chart")
        st.image(str(ml_comp_path), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: DL RESULTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🧬 DL Results":
    st.title("🧬 Deep Learning Results")

    dl_results = load_json_results(str(DL_RESULTS_PATH))

    if dl_results is None:
        st.warning(
            "DL results not found (`dl_results.json`). "
            "Please run the `train_dl.ipynb` notebook first."
        )
        st.stop()

    # ── Results table ──────────────────────────────────────────────────
    st.subheader("📊 Model Performance Comparison")
    res_df = pd.DataFrame(dl_results).T
    res_df.index.name = "Model"

    st.dataframe(
        res_df.style.highlight_max(axis=0, color="#2ecc71",
                                   subset=[c for c in res_df.columns if c in
                                           ["accuracy", "precision", "recall", "f1_score", "cv_accuracy_mean"]]),
        use_container_width=True,
    )

    # ── Bar chart ──────────────────────────────────────────────────────
    st.subheader("📈 Metrics Comparison")
    available_metrics = [c for c in res_df.columns if "std" not in c]
    metrics_to_show = st.multiselect(
        "Select metrics:",
        available_metrics,
        default=[m for m in ["accuracy", "f1_score"] if m in available_metrics],
    )

    if metrics_to_show:
        fig, ax = plt.subplots(figsize=(10, 5))
        res_df[metrics_to_show].plot(kind="bar", ax=ax, colormap="plasma", edgecolor="white")
        ax.set_ylabel("Score")
        ax.set_title("Deep Learning Model Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        plt.xticks(rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Saved comparison chart ─────────────────────────────────────────
    dl_comp = STATIC_DIR / "dl_comparison.png"
    if dl_comp.exists():
        st.markdown("---")
        st.subheader("Overall Comparison Chart")
        st.image(str(dl_comp), use_container_width=True)

    st.markdown("---")

    # ── Confusion matrices ─────────────────────────────────────────────
    st.subheader("📉 Confusion Matrices")
    dl_model_names = list(dl_results.keys())
    cm_cols = st.columns(len(dl_model_names))

    for idx, name in enumerate(dl_model_names):
        cm_path = STATIC_DIR / f"cm_{name}.png"
        with cm_cols[idx]:
            if cm_path.exists():
                st.image(str(cm_path), caption=f"{name}", use_container_width=True)
            else:
                st.caption(f"No confusion matrix for {name}")

    # ── Training history ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Training History")
    hist_cols = st.columns(len(dl_model_names))

    for idx, name in enumerate(dl_model_names):
        hist_path = STATIC_DIR / f"dl_history_{name}.png"
        with hist_cols[idx]:
            if hist_path.exists():
                st.image(str(hist_path), caption=f"{name} – Training History", use_container_width=True)
            else:
                st.caption(f"No training history for {name}")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("🔮 Stress Level Prediction")
    st.markdown("Upload an EEG recording file or manually input feature values to predict the stress level.")

    scaler, feature_cols = load_scaler_and_cols()
    if scaler is None or feature_cols is None:
        st.error("Scaler or feature columns not found. Train models first.")
        st.stop()

    # ── Input method ───────────────────────────────────────────────────
    input_method = st.radio("Choose input method:", ["📁 Upload EEG file (.txt)", "✏️ Manual feature input"], horizontal=True)

    input_features: np.ndarray | None = None

    if input_method == "📁 Upload EEG file (.txt)":
        st.markdown(
            "Upload a raw EEG `.txt` file (same format as training data: 25 columns). "
            "Features will be extracted automatically."
        )
        uploaded = st.file_uploader("Upload EEG file", type=["txt", "csv"])
        if uploaded is not None:
            from data_preprocessing import load_single_file, extract_features, SENSOR_COLS
            import tempfile, os

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                raw_df = load_single_file(tmp_path)
                if raw_df.empty:
                    st.error("Could not parse the uploaded file. Ensure it matches the expected EEG format.")
                else:
                    st.success(f"Loaded {len(raw_df)} EEG samples from the file.")
                    feats = extract_features(raw_df)
                    feat_values = np.array([feats.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
                    input_features = scaler.transform(feat_values)

                    # Show extracted features
                    with st.expander("View extracted features"):
                        feat_df = pd.DataFrame([feats])
                        st.dataframe(feat_df, use_container_width=True)
            finally:
                os.unlink(tmp_path)

    else:  # Manual input
        st.markdown("Select a sample from the dataset or enter features manually.")

        df = load_dataset()
        if df is not None:
            use_sample = st.checkbox("Use a sample from the dataset", value=True)
            if use_sample:
                sample_idx = st.number_input("Sample index:", min_value=0,
                                              max_value=len(df) - 1, value=0, step=1)
                sample = df.iloc[sample_idx]
                feat_values = sample[feature_cols].values.astype(float).reshape(1, -1)
                input_features = scaler.transform(feat_values)

                actual_label = STRESS_MAP.get(int(sample["stress_level"]), "Unknown")
                st.info(f"**Actual label:** {actual_label} | **Task:** {sample['task']} | **Participant:** {int(sample['participant'])}")

                with st.expander("View feature values"):
                    st.dataframe(pd.DataFrame(feat_values, columns=feature_cols), use_container_width=True)
            else:
                st.info("Enter numeric values for each feature below.")
                manual_vals = {}
                cols_per_row = 4
                for i in range(0, len(feature_cols), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(feature_cols[i:i + cols_per_row]):
                        with row_cols[j]:
                            manual_vals[col_name] = st.number_input(col_name, value=0.0, format="%.6f", key=col_name)
                feat_values = np.array([manual_vals[c] for c in feature_cols]).reshape(1, -1)
                input_features = scaler.transform(feat_values)

    # ── Run prediction ─────────────────────────────────────────────────
    if input_features is not None:
        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        model_type = st.radio("Select model type:", ["Deep Learning", "Machine Learning"], horizontal=True)

        if model_type == "Deep Learning":
            if not TORCH_AVAILABLE:
                st.error("PyTorch is not installed. Install it to use DL models.")
                st.stop()

            dl_model_name = st.selectbox("Select DL model:", ["DNN", "CNN_1D", "LSTM"])
            model = load_dl_model(dl_model_name, len(feature_cols))

            if model is None:
                st.error(f"Model `dl_{dl_model_name}.pt` not found in models/.")
            else:
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(input_features)
                    output = model(x_tensor)
                    probs = torch.softmax(output, dim=1).numpy()[0]
                    pred_class = int(np.argmax(probs))

                pred_label = STRESS_MAP[pred_class]
                pred_color = STRESS_COLORS[pred_label]

                # Display result
                result_col1, result_col2 = st.columns([1, 2])
                with result_col1:
                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:20px; border-radius:10px;
                                    background-color:{pred_color}22; border: 2px solid {pred_color};">
                            <h2 style="color:{pred_color}; margin:0;">{pred_label}</h2>
                            <p style="font-size:1.2em; margin:5px 0;">Confidence: {probs[pred_class]*100:.1f}%</p>
                            <p style="margin:0;">Model: {dl_model_name}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with result_col2:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    bars = ax.barh(STRESS_NAMES, probs, color=[STRESS_COLORS[n] for n in STRESS_NAMES],
                                   edgecolor="white")
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probability")
                    ax.set_title("Class Probabilities")
                    for bar, prob in zip(bars, probs):
                        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                                f"{prob:.3f}", va="center", fontsize=10)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        else:  # Machine Learning
            ml_model_names = ["RandomForest", "SVM", "KNN", "GradientBoosting", "LogisticRegression"]
            ml_model_name = st.selectbox("Select ML model:", ml_model_names)
            model = load_ml_model(ml_model_name)

            if model is None:
                st.error(
                    f"Model `ml_{ml_model_name}.pkl` not found. "
                    "Run the `train_ml.ipynb` notebook first."
                )
            else:
                pred_class = int(model.predict(input_features)[0])
                pred_label = STRESS_MAP[pred_class]
                pred_color = STRESS_COLORS[pred_label]

                # Probabilities (if the model supports it)
                probs = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_features)[0]

                result_col1, result_col2 = st.columns([1, 2])
                with result_col1:
                    conf_text = f"Confidence: {probs[pred_class]*100:.1f}%" if probs is not None else ""
                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:20px; border-radius:10px;
                                    background-color:{pred_color}22; border: 2px solid {pred_color};">
                            <h2 style="color:{pred_color}; margin:0;">{pred_label}</h2>
                            <p style="font-size:1.2em; margin:5px 0;">{conf_text}</p>
                            <p style="margin:0;">Model: {ml_model_name}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with result_col2:
                    if probs is not None:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        bars = ax.barh(STRESS_NAMES, probs,
                                       color=[STRESS_COLORS[n] for n in STRESS_NAMES],
                                       edgecolor="white")
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Probability")
                        ax.set_title("Class Probabilities")
                        for bar, prob in zip(bars, probs):
                            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                                    f"{prob:.3f}", va="center", fontsize=10)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("This model does not provide probability estimates.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • EEG Cognitive Load Detection")
