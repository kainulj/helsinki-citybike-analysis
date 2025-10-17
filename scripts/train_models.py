"""
Model training for Helsinki city bike demand prediction.

This script trains two models:
1. A LightGBM regressor for direct demand prediction.
2. A two-phase model (LightGBM classifier + regressor) that first predicts
   if demand > 0, and then predicts the demand amount.

Both models are evaluated and saved.

Note:
    Features and hyperparameters are hardcoded, reflecting the results of SHAP
    analysis and tuning from 'notebooks/03_predict_hourly_departures.ipynb'.

Command-line arguments:
    --train-data (str): Path to the train feature data CSV file.
    --output-dir (str): Path to the directory where trained models will be saved.

Example:
    python feature_engineering.py \
        --train-data data/processed/train.csv \
        --output-dir models \
"""
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
import os
import argparse
import joblib
from citybike.io_utils import load_csv

# Features selected based on mean absolute SHAP values from the LightGBM regressor
SELECTED_FEATURES = [
    'dep_same_hour_mean_7d', 'dep_lag_1', 'hour', 'dep_roll_mean_3', 'dep_lag_2',
    'station_id', 'temperature', 'dep_lag_24', 'dep_lag_168', 'is_weekend',
    'month', 'dep_roll_mean_24', 'weekday', 'precipitation', 'dep_roll_std_24',
    'year', 'dep_lag_72', 'dep_same_hour_std_7d', 'dep_lag_3', 'wind_speed'
]

def train_lightgbm_regressor(X, y):
    """
    Train a single LightGBM regressor.
    """
    model= LGBMRegressor(
        # Tuned hyperparameters
        random_state=0,
        verbose=-1,
        bagging_fraction=np.float64(0.9057331994746333),
        feature_fraction=np.float64(0.5892686656410099),
        learning_rate=np.float64(0.04479216014934237),
        max_depth=12,
        min_child_samples=50,
        n_estimators=350,
        num_leaves=127
    )
    model.fit(X, y)

    return model

def train_two_phase_model(X, y):
    """
    Train a two-phase model (classifier + regressor).
    """
    pos_inds = y > 0

    # Train the classifier model 
    model_clf = LGBMClassifier(
        # Tuned hyperparameters
        random_state=0,
        verbose=-1,
        bagging_fraction=np.float64(0.8442905768097371),
        feature_fraction=np.float64(0.5163806976887504),
        learning_rate=np.float64(0.04238748152900556),
        max_depth=8,
        min_child_samples=38,
        n_estimators=500,
        num_leaves=111
    )
    model_clf.fit(X, (y > 0).astype(int))

    # Train the regression model only on the non-zero values
    model_reg = LGBMRegressor(
        # Tuned hyperparameters
        random_state=0,
        verbose=-1,
        bagging_fraction=np.float64(0.8177888306580715),
        feature_fraction=np.float64(0.5362878147369727),
        learning_rate=np.float64(0.06413085337918903),
        max_depth=10,
        min_child_samples=28,
        n_estimators=375,
        num_leaves=100
    )
    model_reg.fit(X[pos_inds], y[pos_inds])

    return model_clf, model_reg


def main(train_data, output_dir):
    """
    Train, evaluate and save the models.
    """
    dtypes = {'hour': 'category', 'weekday': 'category', 'month': 'category', 'station_id': 'category'}
    try:
        train_df = load_csv(train_data, dtype=dtypes)
    except FileNotFoundError as e:
        print(e)
        return
    
    target_col = 'departures'
    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df[target_col]

    print("Training LightGBM Regressor")
    lgbm_model = train_lightgbm_regressor(X_train, y_train)

    print("Training Two-Phase Model")
    clf, reg = train_two_phase_model(X_train, y_train)

    # Save Lightgbm regressor
    joblib.dump(lgbm_model, os.path.join(output_dir, "lgbm_regressor.pkl"))

    # Save two-phase models
    joblib.dump(clf, os.path.join(output_dir, "two_phase_classifier.pkl"))
    joblib.dump(reg, os.path.join(output_dir, "two_phase_regressor.pkl"))

    print(f"Models saved to {output_dir}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--train-data', type=str, default='data/processed/train.csv', help='Input train data CSV file')
    args.add_argument('--output-dir', type=str, default='models', help='Path to the directory where trained models will be saved.')

    args = args.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args.train_data, args.output_dir)