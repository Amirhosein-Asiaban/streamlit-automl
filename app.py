from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import streamlit as st

# ----------------------------- Configuration -----------------------------

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
PYCARET_SESSION_ID = 42
PYCARET_VERBOSE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionKeys:
    PROBLEM_TYPE = "problem_type"
    TRAINING_DATA = "training_data"
    TRAINING_FILE_FINGERPRINT = "training_file_fingerprint"
    TARGET_COLUMN = "target_column"
    TRAINED_FEATURES = "trained_features"
    BEST_MODEL = "best_model"
    LEADERBOARD = "leaderboard"
    MODEL_TRAINED = "model_trained"
    PREDICTION_DATA = "prediction_data"
    PREDICTION_FILE_FINGERPRINT = "prediction_file_fingerprint"
    IDENTIFIER_COLUMN = "identifier_column"
    PREDICTION_RESULT = "prediction_result"
    TRAINING_ERROR = "training_error"
    PREDICTION_ERROR = "prediction_error"


DEFAULTS = {
    SessionKeys.PROBLEM_TYPE: "Regression",
    SessionKeys.TRAINING_DATA: None,
    SessionKeys.TRAINING_FILE_FINGERPRINT: None,
    SessionKeys.TARGET_COLUMN: None,
    SessionKeys.TRAINED_FEATURES: None,
    SessionKeys.BEST_MODEL: None,
    SessionKeys.LEADERBOARD: None,
    SessionKeys.MODEL_TRAINED: False,
    SessionKeys.PREDICTION_DATA: None,
    SessionKeys.PREDICTION_FILE_FINGERPRINT: None,
    SessionKeys.IDENTIFIER_COLUMN: None,
    SessionKeys.PREDICTION_RESULT: None,
    SessionKeys.TRAINING_ERROR: None,
    SessionKeys.PREDICTION_ERROR: None,
}


# ------------------------------- State -----------------------------------

def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def reset_model_and_prediction() -> None:
    st.session_state[SessionKeys.BEST_MODEL] = None
    st.session_state[SessionKeys.LEADERBOARD] = None
    st.session_state[SessionKeys.MODEL_TRAINED] = False
    st.session_state[SessionKeys.TRAINED_FEATURES] = None
    st.session_state[SessionKeys.TRAINING_ERROR] = None
    reset_prediction()


def reset_prediction() -> None:
    st.session_state[SessionKeys.PREDICTION_DATA] = None
    st.session_state[SessionKeys.PREDICTION_FILE_FINGERPRINT] = None
    st.session_state[SessionKeys.IDENTIFIER_COLUMN] = None
    st.session_state[SessionKeys.PREDICTION_RESULT] = None
    st.session_state[SessionKeys.PREDICTION_ERROR] = None


def reset_for_training_file() -> None:
    st.session_state[SessionKeys.TRAINING_DATA] = None
    st.session_state[SessionKeys.TRAINING_FILE_FINGERPRINT] = None
    st.session_state[SessionKeys.TARGET_COLUMN] = None
    reset_model_and_prediction()


def reset_for_problem_type() -> None:
    reset_for_training_file()


def reset_for_target() -> None:
    reset_model_and_prediction()


# ------------------------------- Data ------------------------------------

def fingerprint(uploaded_file: Any) -> str:
    """Compute a stable fingerprint for the uploaded file."""
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    return f"{uploaded_file.name}:{len(content)}:{digest}"


@st.cache_data(show_spinner=False)
def _read_csv_cached(name: str, content: bytes) -> pd.DataFrame:
    """Cache CSV parsing keyed by name + content hash."""
    from io import BytesIO
    return pd.read_csv(BytesIO(content))


def read_csv(uploaded_file: Any) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    size = getattr(uploaded_file, "size", None)
    if size is None:
        size = len(uploaded_file.getvalue())
    if size > MAX_UPLOAD_SIZE_BYTES:
        st.error(
            f"File '{uploaded_file.name}' exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
        )
        return None
    try:
        content = uploaded_file.getvalue()
        data = _read_csv_cached(uploaded_file.name, content)
        if data.empty:
            st.error("The CSV file is empty.")
            return None
        return data
    except Exception as exc:
        logger.exception("Failed to read CSV")
        st.error(f"The CSV file could not be loaded: {exc}")
        return None


def cast_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure column names are plain strings (PyCaret requirement)."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


# ------------------------------- PyCaret ---------------------------------

def _get_pycaret_modules(problem_type: str):
    if problem_type == "Regression":
        from pycaret.regression import setup, compare_models, pull, predict_model
    else:
        from pycaret.classification import setup, compare_models, pull, predict_model
    return setup, compare_models, pull, predict_model


def train(data: pd.DataFrame, target: str, problem_type: str):
    """Train and compare models. Returns (best_model, leaderboard, error)."""
    try:
        setup, compare_models, pull, _ = _get_pycaret_modules(problem_type)

        # PyCaret needs string column names and clean data
        data = cast_columns_to_str(data)
        if target not in data.columns:
            return None, None, f"Target column '{target}' not found."

        # Drop rows where target is missing
        data = data.dropna(subset=[target]).reset_index(drop=True)
        if data.empty:
            return None, None, "No rows left after dropping missing targets."

        setup(
            data=data,
            target=target,
            session_id=PYCARET_SESSION_ID,
            verbose=PYCARET_VERBOSE,
            html=False,
        )
        best_model = compare_models(turbo=True, errors="ignore")
        leaderboard = pull()

        if best_model is None:
            return None, None, "PyCaret could not find a suitable model."
        if not isinstance(leaderboard, pd.DataFrame) or leaderboard.empty:
            return None, None, "Leaderboard is empty."

        return best_model, leaderboard.copy(deep=True), None
    except Exception as exc:
        logger.exception("Training failed")
        return None, None, f"{type(exc).__name__}: {exc}"


def predict(
    model: Any,
    data: pd.DataFrame,
    identifier: str | None,
    problem_type: str,
    trained_features: list[str] | None,
):
    """Run prediction. Returns (result_df, error)."""
    try:
        if model is None:
            return None, "No trained model available."

        data = cast_columns_to_str(data)

        # --- Identifier handling ---
        if identifier is None or identifier not in data.columns:
            # No identifier: create an auto-increment index
            identifiers = pd.Series(
                range(1, len(data) + 1), name="row_id"
            )
            identifier_name = "row_id"
            features = data.copy()
        else:
            identifiers = data[identifier].copy()
            identifier_name = identifier
            features = data.drop(columns=[identifier]).copy()

        # Align features with what the model was trained on
        if trained_features:
            missing = [c for c in trained_features if c not in features.columns]
            if missing:
                return None, (
                    "Prediction CSV is missing required feature columns: "
                    + ", ".join(missing)
                )
            features = features[trained_features]

        _, _, _, predict_model = _get_pycaret_modules(problem_type)
        predictions = predict_model(model, data=features)

        # PyCaret column name compatibility
        label_col = None
        for candidate in ("prediction_label", "Label", "prediction"):
            if candidate in predictions.columns:
                label_col = candidate
                break
        if label_col is None:
            return None, (
                "Could not find a prediction column in the output. "
                f"Available columns: {list(predictions.columns)}"
            )

        if len(predictions) != len(identifiers):
            return None, "Row count mismatch between predictions and input."

        result = pd.DataFrame(
            {
                identifier_name: identifiers.to_numpy(copy=True),
                "prediction": predictions[label_col].to_numpy(copy=True),
            }
        )

        score_col = None
        for candidate in ("prediction_score", "Score"):
            if candidate in predictions.columns:
                score_col = candidate
                break
        if score_col is not None:
            result["score"] = predictions[score_col].to_numpy(copy=True)

        return result, None
    except Exception as exc:
        logger.exception("Prediction failed")
        return None, f"{type(exc).__name__}: {exc}"

# ---------------------------------- UI -----------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AutoML APP",
        page_icon="🤖",
        layout="wide",
    )
    init_state()

    st.title("AutoML APP")
    st.caption("CSV → model comparison → predictions")

    # -------------------- Problem type --------------------
    problem_options = ["Regression", "Classification"]
    current_problem = st.session_state[SessionKeys.PROBLEM_TYPE]
    if current_problem not in problem_options:
        current_problem = "Regression"

    problem_type = st.radio(
        "Problem Type",
        problem_options,
        index=problem_options.index(current_problem),
        horizontal=True,
        key="problem_type_radio",
    )

    if problem_type != st.session_state[SessionKeys.PROBLEM_TYPE]:
        st.session_state[SessionKeys.PROBLEM_TYPE] = problem_type
        reset_for_problem_type()

    # -------------------- Training --------------------
    st.header("1. Training")
    training_file = st.file_uploader(
        "Upload training CSV",
        type=["csv"],
        key="training_uploader",
    )

    if training_file is None:
        if st.session_state[SessionKeys.TRAINING_DATA] is not None:
            reset_for_training_file()
        st.info("Upload a training CSV to continue.")
        return

    current_fingerprint = fingerprint(training_file)
    if current_fingerprint != st.session_state[
        SessionKeys.TRAINING_FILE_FINGERPRINT
    ]:
        data = read_csv(training_file)
        reset_for_training_file()
        if data is not None:
            st.session_state[SessionKeys.TRAINING_FILE_FINGERPRINT] = (
                current_fingerprint
            )
            st.session_state[SessionKeys.TRAINING_DATA] = data
        else:
            # Reading failed; do not cache fingerprint so user can retry
            return

    training_data = st.session_state[SessionKeys.TRAINING_DATA]
    if training_data is None or training_data.empty:
        return

    st.success(
        f"Loaded **{training_data.shape[0]}** rows × "
        f"**{training_data.shape[1]}** columns."
    )

    with st.expander("Preview training data", expanded=False):
        st.dataframe(training_data.head(20), use_container_width=True)

    columns = [str(c) for c in training_data.columns]
    old_target = st.session_state[SessionKeys.TARGET_COLUMN]
    default_index = columns.index(old_target) if old_target in columns else 0
    target = st.selectbox(
        "Target column",
        columns,
        index=default_index,
        key="target_selectbox",
    )

    if old_target != target:
        st.session_state[SessionKeys.TARGET_COLUMN] = target
        if old_target is not None:
            reset_for_target()
        else:
            reset_model_and_prediction()

    if st.button("Compare Models", type="primary"):
        reset_model_and_prediction()
        with st.spinner("Training and comparing models... this may take a while."):
            best_model, leaderboard, error = train(
                training_data,
                target,
                problem_type,
            )

        if error:
            st.session_state[SessionKeys.TRAINING_ERROR] = error
            st.error(f"Training failed: {error}")
        else:
            # Store trained feature names for prediction alignment
            trained_features = [c for c in training_data.columns if c != target]
            trained_features = [str(c) for c in trained_features]

            st.session_state[SessionKeys.BEST_MODEL] = best_model
            st.session_state[SessionKeys.LEADERBOARD] = leaderboard
            st.session_state[SessionKeys.MODEL_TRAINED] = True
            st.session_state[SessionKeys.TRAINED_FEATURES] = trained_features
            st.session_state[SessionKeys.TRAINING_ERROR] = None
            st.success("Model comparison completed successfully.")

    if not st.session_state[SessionKeys.MODEL_TRAINED]:
        return

    leaderboard = st.session_state[SessionKeys.LEADERBOARD]
    if leaderboard is not None:
        st.subheader("Model Leaderboard")
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    # -------------------- Prediction --------------------
    st.header("2. Prediction")
    prediction_file = st.file_uploader(
        "Upload prediction CSV",
        type=["csv"],
        key="prediction_uploader",
    )

    if prediction_file is None:
        if st.session_state[SessionKeys.PREDICTION_DATA] is not None:
            reset_prediction()
        st.info("Upload a prediction CSV to generate predictions.")
        return

    prediction_fingerprint = fingerprint(prediction_file)
    if prediction_fingerprint != st.session_state[
        SessionKeys.PREDICTION_FILE_FINGERPRINT
    ]:
        data = read_csv(prediction_file)
        reset_prediction()
        if data is not None:
            st.session_state[SessionKeys.PREDICTION_FILE_FINGERPRINT] = (
                prediction_fingerprint
            )
            st.session_state[SessionKeys.PREDICTION_DATA] = data
        else:
            return

    prediction_data = st.session_state[SessionKeys.PREDICTION_DATA]
    if prediction_data is None or prediction_data.empty:
        return

    st.success(
        f"Loaded **{prediction_data.shape[0]}** rows × "
        f"**{prediction_data.shape[1]}** columns."
    )

    with st.expander("Preview prediction data", expanded=False):
        st.dataframe(prediction_data.head(20), use_container_width=True)

    prediction_columns = [str(c) for c in prediction_data.columns]

    # Optional identifier: allow "None" so users without an ID column
    # can still get predictions (an auto row_id will be generated).
    identifier_options = ["(None — auto row number)"] + prediction_columns

    old_identifier = st.session_state[SessionKeys.IDENTIFIER_COLUMN]
    if old_identifier in prediction_columns:
        default_idx = identifier_options.index(old_identifier)
    else:
        default_idx = 0

    identifier_choice = st.selectbox(
        "Identifier column (optional)",
        identifier_options,
        index=default_idx,
        key="identifier_selectbox",
        help=(
            "Select a column to use as an identifier in the output. "
            "If your prediction CSV has no ID column, keep the default "
            "and a row number will be generated automatically."
        ),
    )

    # Convert UI choice back to real identifier value
    identifier = (
        None
        if identifier_choice == "(None — auto row number)"
        else identifier_choice
    )

    if old_identifier != identifier:
        st.session_state[SessionKeys.IDENTIFIER_COLUMN] = identifier
        st.session_state[SessionKeys.PREDICTION_RESULT] = None
        st.session_state[SessionKeys.PREDICTION_ERROR] = None

    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            result, error = predict(
                st.session_state[SessionKeys.BEST_MODEL],
                prediction_data,
                identifier,
                st.session_state[SessionKeys.PROBLEM_TYPE],
                st.session_state[SessionKeys.TRAINED_FEATURES],
            )

        if error:
            st.session_state[SessionKeys.PREDICTION_ERROR] = error
            st.session_state[SessionKeys.PREDICTION_RESULT] = None
            st.error(f"Prediction failed: {error}")
        else:
            st.session_state[SessionKeys.PREDICTION_RESULT] = result
            st.session_state[SessionKeys.PREDICTION_ERROR] = None
            st.success("Predictions generated successfully.")

    result = st.session_state[SessionKeys.PREDICTION_RESULT]
    if result is not None:
        st.subheader("Prediction Preview")
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions CSV",
            csv_bytes,
            "predictions.csv",
            "text/csv",
            key="download_predictions",
        )


if __name__ == "__main__":
    main()