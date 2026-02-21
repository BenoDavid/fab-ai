# src/utils.py
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def load_artifacts(model_dir: str = "model"):
    model_path = os.path.join(model_dir, "model.cbm")
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model or metadata not found. Train first.")

    model = CatBoostRegressor()
    model.load_model(model_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta

def build_dataframe_from_payload(payload: Dict, feature_cols: List[str], categorical_cols: List[str], numeric_cols: List[str]) -> pd.DataFrame:
    """
    Accept partial inputs. Missing features are set to NaN.
    """
    row = {}
    for col in feature_cols:
        if col in payload:
            row[col] = payload[col]
        else:
            row[col] = np.nan

    df = pd.DataFrame([row])

    # Coerce dtypes
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in categorical_cols:
        if c in df.columns:
            # Keep as object (string-like), NaN allowed
            if not pd.api.types.is_object_dtype(df[c]):
                df[c] = df[c].astype("object")

    return df