from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

# cms_data.py is located at:
# healthcare-cost-ai/backend/app/ml/cms_data.py
#
# parents[3] takes us back to:
# healthcare-cost-ai/

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "cms"

BENE_2008_PATH = (
    DATA_DIR
    / "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"
)

BENE_2009_PATH = (
    DATA_DIR
    / "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"
)

INPATIENT_PATH = (
    DATA_DIR
    / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
)

OUTPATIENT_PATH = (
    DATA_DIR
    / "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "age",
    "sex",
    "chronic_condition_count",
    "prior_inpatient_visits",
    "prior_outpatient_visits",
    "prior_reimbursement",
]

TARGET_COLUMN = "next_year_cost"


# ============================================================
# CMS CHRONIC CONDITION COLUMNS
# ============================================================

CHRONIC_COLUMNS = [
    "SP_ALZHDMTA",
    "SP_CHF",
    "SP_CHRNKIDN",
    "SP_CNCR",
    "SP_COPD",
    "SP_DEPRESSN",
    "SP_DIABETES",
    "SP_ISCHMCHT",
    "SP_OSTEOPRS",
    "SP_RA_OA",
    "SP_STRKETIA",
]


# ============================================================
# HELPER FUNCTION FOR CMS DATES
# ============================================================

def convert_date(series):
    """
    Convert CMS YYYYMMDD dates into pandas datetime values.
    """

    numeric_dates = pd.to_numeric(
        series,
        errors="coerce"
    )

    return pd.to_datetime(
        numeric_dates.astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )


# ============================================================
# LOAD AND PREPROCESS CMS DATA
# ============================================================

def load_cms_dataset():

    print("=" * 60)
    print("LOADING CMS DE-SynPUF DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Check that all required files exist
    # --------------------------------------------------------

    required_files = [
        BENE_2008_PATH,
        BENE_2009_PATH,
        INPATIENT_PATH,
        OUTPATIENT_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required CMS file not found:\n{file_path}"
            )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print("\nLoading 2008 beneficiary data...")

    bene08 = pd.read_csv(
        BENE_2008_PATH,
        low_memory=False
    )

    print("Loading 2009 beneficiary data...")

    bene09 = pd.read_csv(
        BENE_2009_PATH,
        low_memory=False
    )

    print("Loading inpatient claims...")

    inpatient = pd.read_csv(
        INPATIENT_PATH,
        low_memory=False
    )

    print("Loading outpatient claims...")

    outpatient = pd.read_csv(
        OUTPATIENT_PATH,
        low_memory=False
    )

    print("\nRaw data loaded successfully.")

    print(
        "2008 beneficiaries:",
        f"{len(bene08):,}"
    )

    print(
        "2009 beneficiaries:",
        f"{len(bene09):,}"
    )

    print(
        "Inpatient claim rows:",
        f"{len(inpatient):,}"
    )

    print(
        "Outpatient claim rows:",
        f"{len(outpatient):,}"
    )


    # ========================================================
    # 1. AGE
    # ========================================================

    print("\nCreating age feature...")

    birth_date = convert_date(
        bene08["BENE_BIRTH_DT"]
    )

    reference_date = pd.Timestamp(
        "2008-12-31"
    )

    bene08["age"] = (
        reference_date.year
        - birth_date.dt.year
        - (
            (birth_date.dt.month > reference_date.month)
            |
            (
                (birth_date.dt.month == reference_date.month)
                &
                (birth_date.dt.day > reference_date.day)
            )
        ).astype(int)
    )


    # ========================================================
    # 2. SEX
    # ========================================================

    print("Creating sex feature...")

    # CMS DE-SynPUF:
    # 1 = Male
    # 2 = Female
    #
    # Our application:
    # 1 = Male
    # 0 = Female

    bene08["sex"] = (
        pd.to_numeric(
            bene08["BENE_SEX_IDENT_CD"],
            errors="coerce"
        )
        .map(
            {
                1: 1,
                2: 0,
            }
        )
    )


    # ========================================================
    # 3. CHRONIC CONDITION COUNT
    # ========================================================

    print("Creating chronic condition count...")

    # We inspected the actual CMS file:
    #
    # 1 = condition present
    # 2 = condition not indicated
    #
    # Count how many of the 11 chronic conditions
    # are present for each beneficiary.

    bene08["chronic_condition_count"] = (
        bene08[CHRONIC_COLUMNS]
        .eq(1)
        .sum(axis=1)
    )


    # ========================================================
    # 4. 2008 PRIOR REIMBURSEMENT
    # ========================================================

    print("Creating 2008 prior reimbursement feature...")

    cost_cols = [
        "MEDREIMB_IP",
        "MEDREIMB_OP",
        "MEDREIMB_CAR",
    ]

    for col in cost_cols:

        bene08[col] = pd.to_numeric(
            bene08[col],
            errors="coerce"
        ).fillna(0)

    bene08["prior_reimbursement"] = (
        bene08[cost_cols]
        .sum(axis=1)
    )

    # Some CMS reimbursement records can contain negative
    # adjustment values.
    #
    # For this application's annual cost prediction target,
    # negative annual totals are clipped to zero.

    bene08["prior_reimbursement"] = (
        bene08["prior_reimbursement"]
        .clip(lower=0)
    )


    # ========================================================
    # 5. CREATE 2009 TARGET COST
    # ========================================================

    print("Creating 2009 next-year cost target...")

    for col in cost_cols:

        bene09[col] = pd.to_numeric(
            bene09[col],
            errors="coerce"
        ).fillna(0)

    bene09[TARGET_COLUMN] = (
        bene09[cost_cols]
        .sum(axis=1)
    )

    # Prevent negative annual cost targets.

    bene09[TARGET_COLUMN] = (
        bene09[TARGET_COLUMN]
        .clip(lower=0)
    )

    target = bene09[
        [
            "DESYNPUF_ID",
            TARGET_COLUMN,
        ]
    ].copy()


    # ========================================================
    # 6. 2008 INPATIENT VISITS
    # ========================================================

    print("Calculating 2008 inpatient utilization...")

    inpatient["CLM_FROM_DT"] = convert_date(
        inpatient["CLM_FROM_DT"]
    )

    # IMPORTANT:
    # The file contains 2008-2010 claims.
    # We use ONLY 2008 claims as model input.

    inpatient08 = inpatient[
        inpatient["CLM_FROM_DT"].dt.year == 2008
    ].copy()

    # Count unique claims.
    #
    # Using nunique(CLМ_ID) is safer than counting rows
    # because claims can contain segments.

    inpatient_visits = (
        inpatient08
        .groupby("DESYNPUF_ID")["CLM_ID"]
        .nunique()
        .rename("prior_inpatient_visits")
        .reset_index()
    )


    # ========================================================
    # 7. 2008 OUTPATIENT VISITS
    # ========================================================

    print("Calculating 2008 outpatient utilization...")

    outpatient["CLM_FROM_DT"] = convert_date(
        outpatient["CLM_FROM_DT"]
    )

    outpatient08 = outpatient[
        outpatient["CLM_FROM_DT"].dt.year == 2008
    ].copy()

    outpatient_visits = (
        outpatient08
        .groupby("DESYNPUF_ID")["CLM_ID"]
        .nunique()
        .rename("prior_outpatient_visits")
        .reset_index()
    )


    # ========================================================
    # 8. CREATE BASE PATIENT TABLE
    # ========================================================

    print("Creating patient-level dataset...")

    base = bene08[
        [
            "DESYNPUF_ID",
            "age",
            "sex",
            "chronic_condition_count",
            "prior_reimbursement",
        ]
    ].copy()


    # ========================================================
    # 9. MERGE INPATIENT UTILIZATION
    # ========================================================

    df = base.merge(
        inpatient_visits,
        on="DESYNPUF_ID",
        how="left",
    )


    # ========================================================
    # 10. MERGE OUTPATIENT UTILIZATION
    # ========================================================

    df = df.merge(
        outpatient_visits,
        on="DESYNPUF_ID",
        how="left",
    )


    # ========================================================
    # 11. MERGE 2009 TARGET
    # ========================================================

    # Inner join means we train only on beneficiaries
    # who appear in both 2008 and 2009.

    df = df.merge(
        target,
        on="DESYNPUF_ID",
        how="inner",
    )


    # ========================================================
    # 12. HANDLE PATIENTS WITH NO CLAIMS
    # ========================================================

    # A missing utilization count means the beneficiary
    # had no matching claim in that category in 2008.

    df["prior_inpatient_visits"] = (
        df["prior_inpatient_visits"]
        .fillna(0)
    )

    df["prior_outpatient_visits"] = (
        df["prior_outpatient_visits"]
        .fillna(0)
    )


    # ========================================================
    # 13. CLEAN INVALID VALUES
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )


    # ========================================================
    # 14. AGE FILTER
    # ========================================================

    # This project focuses on Medicare-age beneficiaries.

    df = df[
        (df["age"] >= 65)
        &
        (df["age"] <= 110)
    ].copy()


    # ========================================================
    # 15. CONVERT DATA TYPES
    # ========================================================

    df["prior_inpatient_visits"] = (
        df["prior_inpatient_visits"]
        .astype(int)
    )

    df["prior_outpatient_visits"] = (
        df["prior_outpatient_visits"]
        .astype(int)
    )

    df["chronic_condition_count"] = (
        df["chronic_condition_count"]
        .astype(int)
    )

    df["sex"] = (
        df["sex"]
        .astype(int)
    )


    # ========================================================
    # 16. FINAL MACHINE LEARNING DATASET
    # ========================================================

    final_df = df[
        ["DESYNPUF_ID"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ].copy()

    final_df = final_df.reset_index(
        drop=True
    )


    # ========================================================
    # 17. DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CMS MACHINE LEARNING DATASET CREATED")
    print("=" * 60)

    print(
        "\nNumber of patients:",
        f"{len(final_df):,}"
    )

    print("\nModel Features:")

    for feature in FEATURE_COLUMNS:
        print(" -", feature)

    print(
        "\nPrediction Target:",
        TARGET_COLUMN
    )

    print("\nFirst 5 rows:")

    print(
        final_df.head()
        .to_string(index=False)
    )

    print("\nDataset summary:")

    print(
        final_df[
            FEATURE_COLUMNS + [TARGET_COLUMN]
        ].describe()
    )

    print("\nMissing values:")

    print(
        final_df[
            FEATURE_COLUMNS + [TARGET_COLUMN]
        ].isnull().sum()
    )

    print("\nTarget cost statistics:")

    print(
        "Minimum:",
        round(final_df[TARGET_COLUMN].min(), 2)
    )

    print(
        "Mean:",
        round(final_df[TARGET_COLUMN].mean(), 2)
    )

    print(
        "Median:",
        round(final_df[TARGET_COLUMN].median(), 2)
    )

    print(
        "Maximum:",
        round(final_df[TARGET_COLUMN].max(), 2)
    )

    print("\nCMS preprocessing completed successfully.")

    return final_df


# ============================================================
# RUN DIRECTLY FOR TESTING
# ============================================================

if __name__ == "__main__":

    load_cms_dataset()