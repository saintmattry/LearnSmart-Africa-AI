from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------------------
# LearnSmart Africa AI
# Early student-support decision aid. It must not be used for grading,
# punishment, admissions, or any other high-stakes decision.
# -------------------------------------------------------------------

st.set_page_config(
    page_title="LearnSmart Africa AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/cleaned.csv")
TARGET = "KCSE_band"
TARGET_ORDER = ["SUPPORT", "ON TRACK", "STRONG"]
RANDOM_STATE = 42
TEST_SIZE = 0.20

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
      [data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.18);
        padding: 14px 16px; border-radius: 12px;
      }
      .small-note {font-size: .88rem; opacity: .8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 LearnSmart Africa AI")
st.caption("Early Support Predictor | A teacher-support tool, not a replacement for professional judgement.")

st.info(
    "This prototype estimates a student's historical outcome band from available "
    "school information. Use its result as a prompt for a supportive conversation "
    "and further assessment—not as a final judgement about a student."
)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create preprocessing and classification steps using training columns only."""
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])
