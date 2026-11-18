import pandas as pd

xls = pd.ExcelFile("data/Student Data set.xlsx", engine="openpyxl")
print("Sheets:", xls.sheet_names)

# Read Attributes sheet
attr = pd.read_excel(xls, sheet_name="Attributes", engine="openpyxl")
attr = attr.dropna(axis=1, how='all').dropna(axis=0, how='all')
print(f"\nAttributes shape: {attr.shape}")
print(attr.to_string())

# Now check the target distribution
df = pd.read_excel(xls, sheet_name="Data", engine="openpyxl")
df = df.dropna(axis=1, how='all')
# Drop the junk column
if 'Unnamed: 16375' in df.columns:
    df = df.drop(columns=['Unnamed: 16375'])

print("\n=== KCSEGrade distribution ===")
print(df['KCSEGrade'].value_counts(dropna=False))

print("\n=== F1Grade distribution ===")
print(df['F1Grade'].value_counts(dropna=False))

print("\n=== F2Grade distribution ===")
print(df['F2Grade'].value_counts(dropna=False))

print("\n=== F3Grade distribution ===")
print(df['F3Grade'].value_counts(dropna=False))

print("\n=== MockGrade distribution ===")
print(df['MockGrade'].value_counts(dropna=False))

print("\n=== Absence - describe ===")
print(df['Absence'].describe())
print(df['Absence'].value_counts().head(10))