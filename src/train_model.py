import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load cleaned data
df = pd.read_csv("data/cleaned.csv")
print(f"Loaded: {df.shape}")

# Target
y = df['KCSE_band']

# --- FEATURE SETS ---
# Columns we will NEVER use as inputs
drop_always = ['KCSE_band', 'KCSEGrade']

# EARLY: only pre-Form 4 info
early_drop = drop_always + ['MockGrade', 'F4Subjects', 'Fspecialization', 'Specialization']
X_early = df.drop(columns=early_drop)

# LATE: early + Mock (for comparison)
X_late = df.drop(columns=drop_always + ['F4Subjects', 'Fspecialization', 'Specialization'])

print(f"\nEARLY features ({X_early.shape[1]}): {list(X_early.columns)}")
print(f"LATE features ({X_late.shape[1]}): includes MockGrade")

def make_pipeline(X, model):
    # Use the X that was actually passed in (EARLY or LATE)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
    print(f"Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

def evaluate(X, y, label):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {label}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train distribution:\n{y_train.value_counts(normalize=True).round(3)}")
    print(f"Test distribution:\n{y_test.value_counts(normalize=True).round(3)}")

    models = {
        'BASELINE (Most Frequent)': DummyClassifier(strategy='most_frequent'),
        'MODEL A - Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'MODEL B - Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1),
    }

    for name, model in models.items():
        print(f"\n--- {name} ---")
        pipe = make_pipeline(X, model)
        # Cross-validation on train only (5-fold)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro')
        print(f"CV Macro F1 (train): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Train on full train, test on held-out
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(f"TEST Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred, digits=3))
        print("Confusion Matrix (rows=True, cols=Predicted):")
        print(f"Labels order: {sorted(y.unique())}")
        print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

# Run both
evaluate(X_early, y, "EARLY MODEL - No MockGrade (Main Model)")
evaluate(X_late, y, "LATE MODEL - With MockGrade (Comparison)")