from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
SOURCE_FILE = DATA_DIR / "Student Data set.xlsx"
OUTPUT_FILE = DATA_DIR / "cleaned.csv"

GRADE_COLUMNS = ["F1Grade", "F2Grade", "F3Grade", "MockGrade", "KCSEGrade"]

# Columns that are unavailable at the intended prediction point, may leak the
# outcome, or were intentionally excluded from this prototype.
LEAKAGE_COLUMNS = [
    "Type_of_Institute",
    "EChallenge", "LEffect", "AEffect",
    "ECAbsence", "ETAbsence", "EFsyllabus", "ECActivities",
    "Edrugs", "ECExam", "ERModel",
    "ETlab", "ELibrary", "EClab", "EElectricity", "EInternet",
]

# Sensitive attributes excluded from model predictors in this prototype.
SENSITIVE_COLUMNS = ["Gender", "Religion", "Disability", "County"]


def to_band(grade):
    """Map a valid KCSE letter grade to the project's three outcome bands."""
    if pd.isna(grade):
        return np.nan
    grade = str(grade).strip().upper()
    if grade in {"A", "B"}:
        return "STRONG"
    if grade == "C":
        return "ON TRACK"
    if grade in {"D", "E"}:
        return "SUPPORT"
    return np.nan


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE_FILE}. Put the original Excel workbook "
            "inside the project's data/ folder."
        )

    workbook = pd.ExcelFile(SOURCE_FILE, engine="openpyxl")
    if "Data" not in workbook.sheet_names:
        raise ValueError("The workbook does not contain a sheet named 'Data'.")

    df = pd.read_excel(workbook, sheet_name="Data", engine="openpyxl")
    df = df.dropna(axis=1, how="all")

    # Remove an accidental empty Excel column if it exists.
    junk_columns = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if junk_columns:
        df = df.drop(columns=junk_columns)

    print(f"Original dataset shape: {df.shape}")

    if "KCSEGrade" not in df.columns:
        raise ValueError("The Data sheet must contain a 'KCSEGrade' column.")

    # Normalize grades without converting missing values into the string 'NAN'.
    for column in GRADE_COLUMNS:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.upper()
                .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
            )

    before = len(df)
    df = df.dropna(subset=["KCSEGrade"]).copy()
    print(f"Rows without KCSE outcome removed: {before - len(df)}")

    df["KCSE_band"] = df["KCSEGrade"].map(to_band)
    invalid_targets = int(df["KCSE_band"].isna().sum())
    if invalid_targets:
        print(f"Rows with unsupported KCSE grades removed: {invalid_targets}")
        df = df.dropna(subset=["KCSE_band"]).copy()

    # Do not impute predictors here. Imputation belongs inside the sklearn
    # Pipeline so medians/modes are learned from training data only.
    columns_to_drop = [
        column
        for column in LEAKAGE_COLUMNS + SENSITIVE_COLUMNS
        if column in df.columns
    ]
    df = df.drop(columns=columns_to_drop)

    if df.empty:
        raise ValueError("No usable rows remain after cleaning and target creation.")

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nTarget distribution:")
    print(df["KCSE_band"].value_counts())
    print("\nTarget distribution (%):")
    print((df["KCSE_band"].value_counts(normalize=True) * 100).round(1))
    print(f"\nDropped leakage/sensitive columns: {columns_to_drop}")
    print(f"Saved cleaned dataset: {OUTPUT_FILE}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
