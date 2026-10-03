import json

import joblib
import numpy as np
import pandas as pd

from .cms_data import FEATURE_COLUMNS
from .train import (
    MODEL_PATH,
    METRICS_PATH,
    SCALER_PATH,
    train_models,
)


class CostPredictor:
    def __init__(self):

        # ====================================================
        # TRAIN MODEL IF ARTIFACTS DO NOT EXIST
        # ====================================================

        if not MODEL_PATH.exists() or not METRICS_PATH.exists():
            print("Model artifacts not found.")
            print("Training model using CMS DE-SynPUF dataset...")

            train_models()

        # ====================================================
        # LOAD TRAINED MODEL
        # ====================================================

        self.model = joblib.load(MODEL_PATH)

        # ====================================================
        # LOAD SCALER
        # ====================================================

        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
        else:
            self.scaler = None

        # ====================================================
        # LOAD MODEL METRICS
        # ====================================================

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            self.metrics = json.load(file)

        # ====================================================
        # BEST MODEL
        # ====================================================

        self.best_model_name = self.metrics["best_model"]

        # Linear Regression was trained using StandardScaler.
        # Random Forest and Gradient Boosting were not.
        self.needs_scaling = (
            self.best_model_name == "Linear Regression"
        )

        # ====================================================
        # RISK THRESHOLDS
        # ====================================================

        self.medium_threshold = float(
            self.metrics["medium_threshold"]
        )

        self.high_threshold = float(
            self.metrics["high_threshold"]
        )

        print(
            f"Loaded model: {self.best_model_name}"
        )

        print(
            f"Medium-risk threshold: "
            f"${self.medium_threshold:,.2f}"
        )

        print(
            f"High-risk threshold: "
            f"${self.high_threshold:,.2f}"
        )

    # ========================================================
    # VECTORIZE INPUT
    # ========================================================

    def _vectorize(
        self,
        features: dict
    ):

        # Check that all required CMS model features exist.
        missing_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in features
        ]

        if missing_features:
            raise ValueError(
                "Missing required prediction features: "
                + ", ".join(missing_features)
            )

        # Keep exactly the same feature names and order
        # used during model training.
        row = {
            column: features[column]
            for column in FEATURE_COLUMNS
        }

        # Use a DataFrame instead of a NumPy array.
        # This preserves feature names and prevents the
        # sklearn "X does not have valid feature names" warning.
        return pd.DataFrame(
            [row],
            columns=FEATURE_COLUMNS
        )

    # ========================================================
    # PREDICT HEALTHCARE COST
    # ========================================================

    def predict(
        self,
        features: dict
    ) -> dict:

        # Convert input dictionary to model-ready DataFrame.
        X = self._vectorize(
            features
        )

        # ====================================================
        # APPLY SCALING IF REQUIRED
        # ====================================================

        if (
            self.needs_scaling
            and self.scaler is not None
        ):
            X_input = self.scaler.transform(X)

        else:
            X_input = X

        # ====================================================
        # MAKE COST PREDICTION
        # ====================================================

        prediction = float(
            self.model.predict(
                X_input
            )[0]
        )

        # Predicted healthcare cost should not be negative.
        prediction = max(
            prediction,
            0.0
        )

        # ====================================================
        # DETERMINE RISK LEVEL
        # ====================================================

        if prediction >= self.high_threshold:
            risk = "high"

        elif prediction >= self.medium_threshold:
            risk = "medium"

        else:
            risk = "low"

        # ====================================================
        # ESTIMATE PERCENTILE
        # ====================================================

        percentile = self._approx_percentile(
            prediction
        )

        # ====================================================
        # CREATE RESOURCE PLANNING RECOMMENDATION
        # ====================================================

        recommendation = self._recommendation(
            risk,
            features
        )

        # ====================================================
        # RETURN API RESULT
        # ====================================================

        return {
            "predicted_cost": round(
                prediction,
                2
            ),
            "risk_level": risk,
            "percentile": percentile,
            "recommendation": recommendation,
        }

    # ========================================================
    # APPROXIMATE COST PERCENTILE
    # ========================================================

    def _approx_percentile(
        self,
        prediction: float
    ) -> int:

        # ----------------------------------------------------
        # HIGH RISK
        # ----------------------------------------------------

        if prediction >= self.high_threshold:

            upper_bound = (
                self.high_threshold * 2.0
            )

            span = max(
                upper_bound
                - self.high_threshold,
                1
            )

            fraction = min(
                (
                    prediction
                    - self.high_threshold
                )
                / span,
                1.0
            )

            percentile = (
                85
                + fraction * 14
            )

            return int(
                min(
                    percentile,
                    99
                )
            )

        # ----------------------------------------------------
        # MEDIUM RISK
        # ----------------------------------------------------

        if prediction >= self.medium_threshold:

            span = max(
                self.high_threshold
                - self.medium_threshold,
                1
            )

            fraction = (
                prediction
                - self.medium_threshold
            ) / span

            percentile = (
                60
                + fraction * 25
            )

            return int(
                min(
                    percentile,
                    84
                )
            )

        # ----------------------------------------------------
        # LOW RISK
        # ----------------------------------------------------

        span = max(
            self.medium_threshold,
            1
        )

        fraction = min(
            prediction / span,
            1.0
        )

        percentile = (
            fraction * 60
        )

        return int(
            percentile
        )

    # ========================================================
    # RESOURCE PLANNING RECOMMENDATION
    # ========================================================

    def _recommendation(
        self,
        risk: str,
        features: dict
    ) -> str:

        inpatient_visits = int(
            features.get(
                "prior_inpatient_visits",
                0
            )
        )

        outpatient_visits = int(
            features.get(
                "prior_outpatient_visits",
                0
            )
        )

        chronic_conditions = int(
            features.get(
                "chronic_condition_count",
                0
            )
        )

        prior_reimbursement = float(
            features.get(
                "prior_reimbursement",
                0
            )
        )

        # ----------------------------------------------------
        # HIGH RISK
        # ----------------------------------------------------

        if risk == "high":

            if inpatient_visits >= 2:

                return (
                    "High expected cost with repeated "
                    "inpatient utilization. Prioritize "
                    "case-management outreach and "
                    "post-discharge follow-up."
                )

            if chronic_conditions >= 5:

                return (
                    "High expected cost with substantial "
                    "chronic-disease burden. Consider "
                    "proactive chronic-care management "
                    "and closer utilization monitoring."
                )

            return (
                "High expected healthcare cost. "
                "Flag for proactive case-management "
                "and resource-planning review."
            )

        # ----------------------------------------------------
        # MEDIUM RISK
        # ----------------------------------------------------

        if risk == "medium":

            if outpatient_visits >= 5:

                return (
                    "Moderate expected cost with elevated "
                    "outpatient utilization. Monitor "
                    "utilization trends and coordinate "
                    "preventive care."
                )

            if prior_reimbursement >= 5000:

                return (
                    "Moderate expected cost with notable "
                    "prior-year reimbursement. Review "
                    "utilization quarterly for emerging "
                    "high-cost patterns."
                )

            return (
                "Moderate expected healthcare cost. "
                "Continue periodic utilization monitoring "
                "and chronic-care follow-up."
            )

        # ----------------------------------------------------
        # LOW RISK
        # ----------------------------------------------------

        return (
            "Low expected healthcare cost. "
            "Continue routine preventive care "
            "and annual utilization monitoring."
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def feature_importance(
        self
    ):

        return self.metrics.get(
            "feature_importance",
            {}
        )

    # ========================================================
    # MODEL METRICS
    # ========================================================

    def model_metrics(
        self
    ):

        return self.metrics


# ============================================================
# SINGLETON PREDICTOR
# ============================================================

_predictor_instance = None


def get_predictor() -> CostPredictor:

    global _predictor_instance

    if _predictor_instance is None:
        _predictor_instance = CostPredictor()

    return _predictor_instance