import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import shap
import warnings
warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/nishapattim05-del/crime-project-data/refs/heads/main/crime_cleaned_old.csv"

# The 14 VIF-selected features from Phase 3 (all VIF < 10)
FEATURES = [
    "prbarr", "prbconv", "prbpris", "avgsen",
    "polpc", "density", "taxpc", "west", "central",
    "urban", "pctmin80", "wcon", "wtuc", "wfed"
]
TARGET = "crmrte"

# Human-readable labels for UI display
FEATURE_LABELS = {
    "prbarr":   "Probability of arrest",
    "prbconv":  "Probability of conviction",
    "prbpris":  "Probability of prison sentence",
    "avgsen":   "Average prison sentence (days)",
    "polpc":    "Police per capita",
    "density":  "Population density",
    "taxpc":    "Tax revenue per capita ($)",
    "west":     "Western region (0/1)",
    "central":  "Central region (0/1)",
    "urban":    "Urban county (0/1)",
    "pctmin80": "% minority population (1980)",
    "wcon":     "Weekly wage – construction ($)",
    "wtuc":     "Weekly wage – transport ($)",
    "wfed":     "Weekly wage – federal ($)",
}

# Slider ranges for what-if simulator (min, max, default, step)
FEATURE_RANGES = {
    "prbarr":   (0.05, 0.80, 0.30, 0.01),
    "prbconv":  (0.05, 1.00, 0.50, 0.01),
    "prbpris":  (0.05, 0.80, 0.40, 0.01),
    "avgsen":   (1.0,  20.0, 9.0,  0.5),
    "polpc":    (0.001, 0.01, 0.002, 0.0001),
    "density":  (0.0,  8.0,  1.0,  0.1),
    "taxpc":    (20.0, 120.0, 40.0, 1.0),
    "west":     (0, 1, 0, 1),
    "central":  (0, 1, 0, 1),
    "urban":    (0, 1, 0, 1),
    "pctmin80": (0.0, 70.0, 20.0, 1.0),
    "wcon":     (150.0, 500.0, 285.0, 5.0),
    "wtuc":     (200.0, 700.0, 375.0, 5.0),
    "wfed":     (300.0, 800.0, 450.0, 5.0),
}


def load_data():
    """Load dataset from GitHub or fall back to synthetic demo data."""
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception:
        return _generate_demo_data()


def _generate_demo_data():
    """
    Generate a realistic synthetic dataset that mirrors the
    Cornwell & Trumbull NC structure when GitHub is unreachable.
    90 counties x 7 years = 630 rows.
    """
    np.random.seed(42)
    n_counties = 90
    years = range(1981, 1988)
    rows = []
    for county in range(1, n_counties + 1):
        base_crime = np.random.uniform(0.01, 0.10)
        for yr in years:
            row = {
                "county":   county,
                "year":     yr,
                "crmrte":   max(0.005, base_crime + np.random.normal(0, 0.005)),
                "prbarr":   np.random.uniform(0.10, 0.70),
                "prbconv":  np.random.uniform(0.10, 0.90),
                "prbpris":  np.random.uniform(0.10, 0.70),
                "avgsen":   np.random.uniform(3, 18),
                "polpc":    np.random.uniform(0.001, 0.008),
                "density":  np.random.uniform(0.01, 7.0),
                "taxpc":    np.random.uniform(25, 110),
                "west":     int(county % 3 == 0),
                "central":  int(county % 3 == 1),
                "urban":    int(np.random.random() > 0.7),
                "pctmin80": np.random.uniform(2, 65),
                "wcon":     np.random.uniform(180, 470),
                "wtuc":     np.random.uniform(220, 650),
                "wfed":     np.random.uniform(320, 750),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_xy(df):
    """Return feature matrix X and target y, dropping rows with nulls."""
    cols = FEATURES + [TARGET]
    sub = df[cols].dropna()
    X = sub[FEATURES]
    y = sub[TARGET]
    return X, y


def train_model(df):
    """
    Train the Random Forest model (mirrors Phase 3 best model, R²≈0.905).
    Returns: model, scaler, X_test, y_test, shap_values, explainer, feature_names
    """
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # SHAP explainability
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sc)

    return {
        "model":         model,
        "scaler":        scaler,
        "X_test":        X_test,
        "y_test":        y_test,
        "shap_values":   shap_values,
        "explainer":     explainer,
        "feature_names": FEATURES,
        "r2":            round(r2, 4),
        "rmse":          round(rmse, 6),
    }


def predict_crime(model_bundle, feature_dict):
    """
    Predict crime rate from a dict of feature values.
    Returns predicted crmrte as a float.
    """
    model   = model_bundle["model"]
    scaler  = model_bundle["scaler"]
    X_input = pd.DataFrame([feature_dict])[FEATURES]
    X_sc    = scaler.transform(X_input)
    return float(model.predict(X_sc)[0])


def get_shap_for_input(model_bundle, feature_dict):
    """Return SHAP values for a single prediction input."""
    explainer = model_bundle["explainer"]
    scaler    = model_bundle["scaler"]
    X_input   = pd.DataFrame([feature_dict])[FEATURES]
    X_sc      = scaler.transform(X_input)
    sv = explainer.shap_values(X_sc)
    # TreeExplainer returns 2D array (n_samples, n_features); flatten safely
    sv_row = sv[0] if sv.ndim == 2 else sv
    return dict(zip(FEATURES, sv_row))


def get_cluster_profiles(df):
    """
    Return summary stats per K-Means cluster label (mirrors Phase 3 k=3 result).
    Falls back to quantile-based pseudo-clusters if 'cluster' col is absent.
    """
    if "cluster" in df.columns:
        return df.groupby("cluster")[FEATURES + [TARGET]].mean().round(4)
    # Pseudo-clusters by crime rate quantile
    df = df.copy()
    df["cluster"] = pd.qcut(df[TARGET], q=3, labels=[0, 1, 2])
    return df.groupby("cluster")[FEATURES + [TARGET]].mean().round(4)
