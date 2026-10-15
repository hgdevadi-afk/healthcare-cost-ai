import os
from google.cloud import storage

UPLOAD_BUCKET = os.environ.get("GCS_UPLOAD_BUCKET")


def upload_csv_to_gcs(data: bytes, job_id: str) -> str:
    """Store a CSV using the VM's attached service account."""
    if not UPLOAD_BUCKET:
        raise RuntimeError("GCS_UPLOAD_BUCKET is not configured")

    client = storage.Client()
    bucket = client.bucket(UPLOAD_BUCKET)

    object_name = f"uploads/{job_id}.csv"
    blob = bucket.blob(object_name)

    blob.upload_from_string(
        data,
        content_type="text/csv",
        if_generation_match=0,
    )

    return f"gs://{UPLOAD_BUCKET}/{object_name}"
