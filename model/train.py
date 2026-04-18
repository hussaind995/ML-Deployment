from __future__ import annotations
import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
import joblib
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "house-prices.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_COL = ["Price"]

MODEL_FEATURES = [
    "SqFt",
    "Bedrooms",
    "Bathrooms",
    "Offers",
    "Brick",
    "Neighborhood"
]

CAT_FEATURES = [
    "Brick",
    "Neighborhood"
]

NUM_FEATURES = [
    "SqFt",
    "Bedrooms",
    "Bathrooms",
    "Offers",
]

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def create_pipeline() -> Pipeline:

    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    num_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median"))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("col_transfromer", column_transformer),
            ("model", model)
        ]
    )

    return pipeline 

def persist_artifacts(model: Pipeline, metrics: dict[str, object]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

def main():
    print("Loading data.....")
    data = load_data(path=DATA_PATH)
    print("Data loaded")

    X, y = data[MODEL_FEATURES], data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE)

    pipeline = create_pipeline()

    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "mse": mse,
        "mae": mae,
        "feats": MODEL_FEATURES,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    }
    persist_artifacts(pipeline, metrics)
    print(f"Saved model artifact to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Validation mse: {mse:.4f}")


if __name__ == "__main__":
    main()


