

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, f1_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


DATA_PATH    = os.path.join("data", "fully_transformed_student_dataset.csv")
MODELS_DIR   = "models"
MODEL_PATH   = os.path.join(MODELS_DIR, "xgb_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

RANDOM_STATE = 42
N_SYNTHETIC  = 3000   


FEATURES = [
    "Age at Enrollment",
    "Average Grade (1st Sem)",
    "Average Grade (2nd Sem)",
    "Unemployment Rate (%)",
    "Inflation Rate (%)",
    "GDP per Capita (USD)",
    "Scholarship Holder",   # 1 = Yes, 0 = No
    "Gender",               # 1 = Male, 0 = Female
    "Debtor",               # 1 = Yes, 0 = No
    "Tuition Fees Up to Date",  # 1 = Yes, 0 = No
    "grade_trend",          # engineered: 2nd - 1st sem grade
    "sem_performance_ratio",# engineered: 2nd / 1st sem grade
]


#
def load_real_data(path: str) -> pd.DataFrame:
    """Load and preprocess the real student dataset."""
    df = pd.read_csv(path)
    print(f"[DATA] Loaded real dataset: {df.shape}")
    print(f"[DATA] Columns: {list(df.columns)}")

    # Map target
    df["Dropout_Binary"] = (df["Student Status"] == "Dropout").astype(int)

    # Categorical columns to encode
    catg_col = df.select_dtypes(include=["object"]).columns.tolist()
    catg_col = [c for c in catg_col if c != "Student Status"]

    # Impute missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    for col in catg_col:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # IQR capping for key numerical columns
    cap_cols = [
        "Age at Enrollment",
        "Average Grade (1st Sem)",
        "Average Grade (2nd Sem)",
        "Unemployment Rate (%)",
        "Inflation Rate (%)",
        "GDP per Capita (USD)",
    ]
    for col in cap_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Feature engineering
    if "Average Grade (1st Sem)" in df.columns and "Average Grade (2nd Sem)" in df.columns:
        df["grade_trend"] = df["Average Grade (2nd Sem)"] - df["Average Grade (1st Sem)"]
        df["sem_performance_ratio"] = np.where(
            df["Average Grade (1st Sem)"] > 0,
            df["Average Grade (2nd Sem)"] / df["Average Grade (1st Sem)"],
            1,
        )

    # Encode categoricals with low cardinality (<=10 unique values)
    for col in catg_col:
        if df[col].nunique() <= 10:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Select numeric features only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "Dropout_Binary"]

    return df[num_cols + ["Dropout_Binary"]]


def generate_synthetic_data(n: int = N_SYNTHETIC, seed: int = RANDOM_STATE) -> pd.DataFrame:
    
    rng = np.random.default_rng(seed)
    print(f"[DATA] Real dataset not found — generating {n} synthetic samples for demo.")

    age           = rng.integers(17, 55, n).astype(float)
    scholarship   = rng.integers(0, 2, n)
    gender        = rng.integers(0, 2, n)
    debtor        = rng.integers(0, 2, n)
    tuition_ok    = rng.integers(0, 2, n)
    unemp         = rng.uniform(5.0, 18.0, n)
    inflation     = rng.uniform(-0.5, 4.5, n)
    gdp           = rng.uniform(15000, 35000, n)
    grade1        = rng.uniform(8.0, 18.5, n)

    # Grade 2 influenced by grade 1 (realistic correlation)
    grade2 = grade1 + rng.normal(0, 1.2, n)
    grade2 = np.clip(grade2, 0, 20)

    grade_trend  = grade2 - grade1
    sem_ratio    = np.where(grade1 > 0, grade2 / grade1, 1.0)

    # Dropout probability driven by domain logic
    dropout_score = (
        - 0.15 * grade1
        - 0.12 * grade2
        + 0.04 * (age - 25)
        + 0.05 * unemp
        - 0.20 * tuition_ok
        - 0.10 * scholarship
        + 0.12 * debtor
        + rng.normal(0, 0.3, n)
    )
    dropout_prob = 1 / (1 + np.exp(-dropout_score))
    dropout      = (rng.uniform(0, 1, n) < dropout_prob).astype(int)

    df = pd.DataFrame({
        "Age at Enrollment":       age,
        "Average Grade (1st Sem)": grade1,
        "Average Grade (2nd Sem)": grade2,
        "Unemployment Rate (%)":   unemp,
        "Inflation Rate (%)":      inflation,
        "GDP per Capita (USD)":    gdp,
        "Scholarship Holder":      scholarship.astype(float),
        "Gender":                  gender.astype(float),
        "Debtor":                  debtor.astype(float),
        "Tuition Fees Up to Date": tuition_ok.astype(float),
        "grade_trend":             grade_trend,
        "sem_performance_ratio":   sem_ratio,
        "Dropout_Binary":          dropout,
    })
    return df



def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    if os.path.exists(DATA_PATH):
        df = load_real_data(DATA_PATH)
  
        available = [f for f in FEATURES if f in df.columns]
        X = df[available]
        y = df["Dropout_Binary"]
        feature_names = available
    else:
        df = generate_synthetic_data()
        X = df[FEATURES]
        y = df["Dropout_Binary"]
        feature_names = FEATURES

    print(f"\n[INFO] Dataset shape : {X.shape}")
    print(f"[INFO] Dropout rate  : {y.mean() * 100:.1f}%")
    print(f"[INFO] Features used : {feature_names}\n")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 4. XGBoost model
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_test_sc, y_test)],
        verbose=False,
    )

    
    y_pred      = model.predict(X_test_sc)
    y_proba     = model.predict_proba(X_test_sc)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred, average="weighted")
    roc_auc     = roc_auc_score(y_test, y_proba)

    print("=" * 50)
    print("  MODEL EVALUATION (Test Set)")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print("=" * 50)

   
    joblib.dump(model,        MODEL_PATH)
    joblib.dump(scaler,       SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)

    print(f"\n[SAVED] Model        : {MODEL_PATH}")
    print(f"[SAVED] Scaler       : {SCALER_PATH}")
    print(f"[SAVED] Feature names: {FEATURES_PATH}")
    print("\n[DONE] Training complete. Run: streamlit run app.py")


if __name__ == "__main__":
    train()
