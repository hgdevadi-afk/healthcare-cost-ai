import asyncio
import io
import uuid
from datetime import datetime

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .. import schemas
from ..database import Job, Patient, SessionLocal, get_db
from ..cloud_storage import upload_csv_to_gcs
from ..ml.cms_data import FEATURE_COLUMNS
from ..ml.predictor import get_predictor


router = APIRouter(
    prefix="/api/upload",
    tags=["upload"],
)


BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 0.35


# ============================================================
# BACKGROUND BULK PROCESSING
# ============================================================

async def process_bulk_job(
    job_id: str,
    records: list[dict],
):
    """
    Simulates the project cloud architecture:

    CSV Upload
        -> Cloud Storage
        -> Pub/Sub
        -> Cloud Run workers
        -> ML model
        -> Database

    Locally, each batch represents a group of messages
    processed asynchronously by a Cloud Run worker.
    """

    db = SessionLocal()
    predictor = get_predictor()

    try:
        job = (
            db.query(Job)
            .filter(Job.id == job_id)
            .first()
        )

        if not job:
            return

        job.status = "processing"
        db.commit()

        total = len(records)
        processed = 0
        high_risk = 0

        # ====================================================
        # PROCESS RECORDS IN BATCHES
        # ====================================================

        for start in range(
            0,
            total,
            BATCH_SIZE,
        ):
            batch = records[
                start : start + BATCH_SIZE
            ]

            # Simulates Pub/Sub -> Cloud Run latency.
            await asyncio.sleep(
                BATCH_DELAY_SECONDS
            )

            for row in batch:

                try:
                    features = (
                        schemas.PatientFeatures
                        .model_validate(row)
                        .model_dump()
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ):
                    continue

                # ============================================
                # MODEL PREDICTION
                # ============================================

                result = predictor.predict(
                    features
                )

                if (
                    result["risk_level"]
                    == "high"
                ):
                    high_risk += 1

                # ============================================
                # SAVE PATIENT RESULT
                # ============================================

                patient = Patient(
                    source=job_id,

                    age=features["age"],
                    sex=features["sex"],

                    chronic_condition_count=features[
                        "chronic_condition_count"
                    ],

                    prior_inpatient_visits=features[
                        "prior_inpatient_visits"
                    ],

                    prior_outpatient_visits=features[
                        "prior_outpatient_visits"
                    ],

                    prior_reimbursement=features[
                        "prior_reimbursement"
                    ],

                    predicted_cost=result[
                        "predicted_cost"
                    ],

                    risk_level=result[
                        "risk_level"
                    ],

                    percentile=result[
                        "percentile"
                    ],
                )

                db.add(patient)

                processed += 1

            # ================================================
            # UPDATE JOB PROGRESS
            # ================================================

            job.processed_records = processed
            job.high_risk_found = high_risk

            db.commit()

        # ====================================================
        # COMPLETE JOB
        # ====================================================

        job.status = "completed"
        job.completed_at = datetime.utcnow()

        db.commit()

    except Exception as exc:

        job = (
            db.query(Job)
            .filter(Job.id == job_id)
            .first()
        )

        if job:
            job.status = "failed"
            job.error_message = str(exc)
            db.commit()

    finally:
        db.close()


# ============================================================
# BULK CSV UPLOAD
# ============================================================

@router.post(
    "/bulk",
    response_model=schemas.JobOut,
)
async def upload_bulk(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    # ========================================================
    # FILE TYPE VALIDATION
    # ========================================================

    if (
        not file.filename
        or not file.filename.lower().endswith(".csv")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .csv file.",
        )

    # ========================================================
    # READ CSV
    # ========================================================

    raw = await file.read()

    try:
        df = pd.read_csv(
            io.BytesIO(raw)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not parse the CSV file."
            ),
        ) from exc

    # ========================================================
    # CHECK REQUIRED CMS MODEL FEATURES
    # ========================================================

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV is missing required columns: "
                + ", ".join(missing)
                + ". Expected columns: "
                + ", ".join(FEATURE_COLUMNS)
            ),
        )

    # ========================================================
    # KEEP ONLY MODEL INPUT FEATURES
    # ========================================================

    records = (
        df[FEATURE_COLUMNS]
        .to_dict(
            orient="records"
        )
    )

    if not records:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no data rows.",
        )

    # ========================================================
    # CREATE BULK PROCESSING JOB
    # ========================================================

    job_id = uuid.uuid4().hex[:10]

    # Save the validated CSV to private Cloud Storage first.
    try:
        upload_csv_to_gcs(raw, job_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to store the uploaded CSV.",
        ) from exc

    job = Job(
        id=job_id,
        status="queued",
        filename=file.filename,
        total_records=len(records),
        processed_records=0,
        high_risk_found=0,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    # ========================================================
    # START BACKGROUND PROCESSING
    # ========================================================

    background_tasks.add_task(
        process_bulk_job,
        job_id,
        records,
    )

    return job


# ============================================================
# SAMPLE CSV
# ============================================================

@router.get("/sample-csv")
def sample_csv():
    """
    Returns a sample CSV matching the six CMS model inputs.
    """

    header = ",".join(
        FEATURE_COLUMNS
    )

    rows = [
        "72,0,2,0,4,3200.50",
        "81,1,5,2,12,15400.00",
        "68,0,0,0,1,650.00",
        "89,1,7,3,20,32000.00",
        "75,0,3,1,6,5200.00",
    ]

    content = (
        header
        + "\n"
        + "\n".join(rows)
    )

    return {
        "filename": "sample_patients.csv",
        "content": content,
    }
