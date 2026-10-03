from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.ml.cms_data import (
    load_cms_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


# ============================================================
# ARTIFACT PATHS
# ============================================================

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TRAINING_DATA_PATH = ARTIFACT_DIR / "training_data.csv"


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models():

    print("=" * 70)
    print("HEALTHCARE COST PREDICTION - MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD REAL CMS DATA
    # --------------------------------------------------------

    print("\nLoading CMS dataset...")

    df = load_cms_dataset()

    print(
        "\nTotal CMS patients available:",
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # 2. FEATURES AND TARGET
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    print("\nFeatures used:")

    for feature in FEATURE_COLUMNS:
        print(" -", feature)

    print(
        "\nTarget:",
        TARGET_COLUMN
    )

    # --------------------------------------------------------
    # 3. TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    print(
        "\nTraining samples:",
        f"{len(X_train):,}"
    )

    print(
        "Testing samples:",
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # 4. SCALE DATA FOR LINEAR REGRESSION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # 5. LINEAR REGRESSION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 70)

    linear_model = LinearRegression()

    linear_model.fit(
        X_train_scaled,
        y_train
    )

    linear_predictions = linear_model.predict(
        X_test_scaled
    )

    # Healthcare cost cannot be negative.
    linear_predictions = np.clip(
        linear_predictions,
        0,
        None
    )

    linear_metrics = calculate_metrics(
        y_test,
        linear_predictions
    )

    print(
        "RMSE:",
        round(linear_metrics["rmse"], 2)
    )

    print(
        "MAE:",
        round(linear_metrics["mae"], 2)
    )

    print(
        "R²:",
        round(linear_metrics["r2"], 4)
    )

    # --------------------------------------------------------
    # 6. RANDOM FOREST
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    random_forest = RandomForestRegressor(
        n_estimators=250,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_predictions = random_forest.predict(
        X_test
    )

    rf_predictions = np.clip(
        rf_predictions,
        0,
        None
    )

    rf_metrics = calculate_metrics(
        y_test,
        rf_predictions
    )

    print(
        "RMSE:",
        round(rf_metrics["rmse"], 2)
    )

    print(
        "MAE:",
        round(rf_metrics["mae"], 2)
    )

    print(
        "R²:",
        round(rf_metrics["r2"], 4)
    )

    # --------------------------------------------------------
    # 7. GRADIENT BOOSTING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING GRADIENT BOOSTING")
    print("=" * 70)

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    gradient_boosting.fit(
        X_train,
        y_train
    )

    gb_predictions = gradient_boosting.predict(
        X_test
    )

    gb_predictions = np.clip(
        gb_predictions,
        0,
        None
    )

    gb_metrics = calculate_metrics(
        y_test,
        gb_predictions
    )

    print(
        "RMSE:",
        round(gb_metrics["rmse"], 2)
    )

    print(
        "MAE:",
        round(gb_metrics["mae"], 2)
    )

    print(
        "R²:",
        round(gb_metrics["r2"], 4)
    )

    # --------------------------------------------------------
    # 8. COMPARE MODELS
    # --------------------------------------------------------

    model_results = {
        "Linear Regression": linear_metrics,
        "Random Forest": rf_metrics,
        "Gradient Boosting": gb_metrics,
    }

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame(
        model_results
    ).T

    print(
        comparison_df.round(4)
    )

    # --------------------------------------------------------
    # 9. SELECT BEST MODEL
    # --------------------------------------------------------

    best_model_name = max(
        model_results,
        key=lambda name:
        model_results[name]["r2"]
    )

    print(
        "\nBest model:",
        best_model_name
    )

    if best_model_name == "Linear Regression":

        best_model = linear_model
        best_scaler = scaler
        best_predictions = linear_predictions

    elif best_model_name == "Random Forest":

        best_model = random_forest
        best_scaler = None
        best_predictions = rf_predictions

    else:

        best_model = gradient_boosting
        best_scaler = None
        best_predictions = gb_predictions

    # --------------------------------------------------------
    # 10. FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = {}

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance_values = (
            best_model.feature_importances_
        )

        feature_importance = {
            feature: float(importance)
            for feature, importance in zip(
                FEATURE_COLUMNS,
                importance_values
            )
        }

    elif hasattr(
        best_model,
        "coef_"
    ):

        importance_values = np.abs(
            best_model.coef_
        )

        total = importance_values.sum()

        if total > 0:
            importance_values = (
                importance_values / total
            )

        feature_importance = {
            feature: float(importance)
            for feature, importance in zip(
                FEATURE_COLUMNS,
                importance_values
            )
        }

    feature_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    print("\nFeature importance:")

    for feature, importance in feature_importance.items():

        print(
            f"{feature}: {importance:.4f}"
        )

    # --------------------------------------------------------
    # 11. RISK THRESHOLDS
    # --------------------------------------------------------

    # Generate predictions on training data from best model.

    if best_model_name == "Linear Regression":

        train_predictions = best_model.predict(
            X_train_scaled
        )

    else:

        train_predictions = best_model.predict(
            X_train
        )

    train_predictions = np.clip(
        train_predictions,
        0,
        None
    )

    medium_threshold = float(
        np.percentile(
            train_predictions,
            60
        )
    )

    high_threshold = float(
        np.percentile(
            train_predictions,
            85
        )
    )

    print(
        "\nMedium-risk threshold:",
        round(medium_threshold, 2)
    )

    print(
        "High-risk threshold:",
        round(high_threshold, 2)
    )

    # --------------------------------------------------------
    # 12. SAVE BEST MODEL
    # --------------------------------------------------------

    print("\nSaving trained model...")

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    joblib.dump(
        best_scaler,
        SCALER_PATH
    )

    # --------------------------------------------------------
    # 13. SAVE METRICS
    # --------------------------------------------------------

    metrics_output = {
        "best_model": best_model_name,

        "models": model_results,

        "feature_importance":
            feature_importance,

        "medium_threshold":
            medium_threshold,

        "high_threshold":
            high_threshold,

        "n_train":
            int(len(X_train)),

        "n_test":
            int(len(X_test)),

        "n_total":
            int(len(df)),

        "features":
            FEATURE_COLUMNS,

        "target":
            TARGET_COLUMN,

        "dataset":
            "CMS Medicare DE-SynPUF Sample 1",

        "prediction_year":
            2009,

        "feature_year":
            2008,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # 14. SAVE TRAINING DATA SAMPLE
    # --------------------------------------------------------

    export_df = df[
        ["DESYNPUF_ID"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ].copy()

    export_df.to_csv(
        TRAINING_DATA_PATH,
        index=False
    )

    print(
        "\nTraining dataset saved to:",
        TRAINING_DATA_PATH
    )

    print(
        "Model saved to:",
        MODEL_PATH
    )

    print(
        "Metrics saved to:",
        METRICS_PATH
    )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return metrics_output


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train_models()