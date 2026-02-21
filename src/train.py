# src/train.py
import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_TARGET = "needed_qty"
DEFAULT_CATS = ["po", "style"]
DEFAULT_NUMS = ["total_qty_to_produce", "estimated_fabrics_needed", "requested_fabrics_qty"]

def infer_columns(df: pd.DataFrame, target: str, cats: List[str], nums: List[str]):
    cols_lower = {c.lower(): c for c in df.columns}
    found_target = cols_lower.get(target.lower(), None)
    if not found_target:
        raise ValueError(
            f"Target column '{target}' not found. Found columns: {list(df.columns)}. "
            f"Please ensure the CSV has a column named '{target}'."
        )
    found_cats = [cols_lower[c.lower()] for c in cats if c.lower() in cols_lower]
    found_nums = [cols_lower[c.lower()] for c in nums if c.lower() in cols_lower]
    used_features = found_cats + found_nums
    if len(used_features) == 0:
        raise ValueError(
            "No feature columns found. At least one of the following must exist: "
            f"{cats + nums}"
        )
    return found_target, found_cats, found_nums

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}

def main():
    parser = argparse.ArgumentParser(description="Train CatBoost model to predict needed_qty.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    parser.add_argument("--cat_cols", type=str, nargs="*", default=DEFAULT_CATS, help="Categorical column names.")
    parser.add_argument("--num_cols", type=str, nargs="*", default=DEFAULT_NUMS, help="Numeric column names.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Holdout test fraction.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--iterations", type=int, default=1000, help="CatBoost iterations.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--depth", type=int, default=8, help="Tree depth.")
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0, help="L2 regularization.")
    parser.add_argument("--model_dir", type=str, default="model", help="Where to save model artifacts.")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    # Strip spaces from column names
    df.columns = [c.strip() for c in df.columns]

    # Infer columns (case-insensitive)
    target_col, cat_cols, num_cols = infer_columns(df, args.target, args.cat_cols, args.num_cols)

    # Use only available columns + target
    feat_cols = cat_cols + num_cols
    data = df[feat_cols + [target_col]].copy()

    # Coerce numeric cols to numeric (keep NaN)
    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Train/validation split
    train_df, val_df = train_test_split(
        data, test_size=args.test_size, random_state=args.random_seed
    )

    X_train = train_df[feat_cols]
    y_train = train_df[target_col]
    X_val = val_df[feat_cols]
    y_val = val_df[target_col]

    # Identify categorical feature indices relative to feat_cols order
    cat_feature_indices = [feat_cols.index(c) for c in cat_cols]

    # CatBoost pools (handles missing values natively)
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.random_seed,
        od_type="Iter",            # early stopping
        od_wait=50,
        verbose=100
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Evaluate
    val_pred = model.predict(val_pool)
    metrics = compute_metrics(y_val.values, val_pred)
    print("Validation metrics:", metrics)

    # Save artifacts
    model_path = os.path.join(args.model_dir, "model.cbm")
    meta_path = os.path.join(args.model_dir, "metadata.json")

    model.save_model(model_path)

    metadata = {
        "target": target_col,
        "feature_cols": feat_cols,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()