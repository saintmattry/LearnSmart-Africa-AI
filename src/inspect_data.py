from pathlib import Path

import pandas as pd

SOURCE_FILE = Path("data/Student Data set.xlsx")


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE_FILE}. Put the original Excel workbook "
            "inside the project's data/ folder."
        )

    workbook = pd.ExcelFile(SOURCE_FILE, engine="openpyxl")
    print("Workbook sheets:", workbook.sheet_names)

    if "Attributes" in workbook.sheet_names:
        attributes = pd.read_excel(workbook, sheet_name="Attributes", engine="openpyxl")
        attributes = attributes.dropna(axis=1, how="all").dropna(axis=0, how="all")
        print(f"\nAttributes sheet shape: {attributes.shape}")
        print(attributes.to_string(index=False))
    else:
        print("\nNo 'Attributes' sheet was found.")

    if "Data" not in workbook.sheet_names:
        raise ValueError("The workbook does not contain a sheet named 'Data'.")

    df = pd.read_excel(workbook, sheet_name="Data", engine="openpyxl")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    for column in ["KCSEGrade", "F1Grade", "F2Grade", "F3Grade", "MockGrade"]:
        if column in df.columns:
            print(f"\n=== {column} distribution ===")
            print(df[column].value_counts(dropna=False))

    if "Absence" in df.columns:
        print("\n=== Absence summary ===")
        print(df["Absence"].describe())
        print("\nMost common absence values:")
        print(df["Absence"].value_counts(dropna=False).head(10))

    print("\n=== Dataset structure ===")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
