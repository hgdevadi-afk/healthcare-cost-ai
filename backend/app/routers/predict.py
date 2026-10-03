from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import schemas
from ..database import Patient, get_db
from ..ml.predictor import get_predictor


router = APIRouter(
    prefix="/api/predict",
    tags=["predict"],
)


@router.post(
    "",
    response_model=schemas.PredictionResult,
)
def predict_single(
    features: schemas.PatientFeatures,
    db: Session = Depends(get_db),
):
    # ========================================================
    # LOAD TRAINED PREDICTOR
    # ========================================================

    predictor = get_predictor()

    # ========================================================
    # RUN COST PREDICTION
    # ========================================================

    result = predictor.predict(
        features.model_dump()
    )

    # ========================================================
    # SAVE PREDICTION TO DATABASE
    # ========================================================

    patient = Patient(
        source="manual",

        # Model input features
        age=features.age,
        sex=features.sex,
        chronic_condition_count=features.chronic_condition_count,
        prior_inpatient_visits=features.prior_inpatient_visits,
        prior_outpatient_visits=features.prior_outpatient_visits,
        prior_reimbursement=features.prior_reimbursement,

        # Model outputs
        predicted_cost=result["predicted_cost"],
        risk_level=result["risk_level"],
        percentile=result["percentile"],
    )

    db.add(patient)
    db.commit()
    db.refresh(patient)

    # ========================================================
    # RETURN PREDICTION
    # ========================================================

    return result