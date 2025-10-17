"""
Model evaluation and visualization for Helsinki city bike demand prediction.

This script loads trained models, computes evaluation metrics,
and generates comparison plots and SHAP visualizations.

Note:
    Model file paths are hardcoded for the final trained models stored in 'models/'.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import metrics
import shap
import os
import joblib
import lightgbm as lgbm
from citybike.io_utils import load_csv
from citybike.evaluation import evaluate

# Style settings
sns.set_theme(context='paper',style='white', font_scale=1.2)
palette = sns.color_palette("Dark2")
sns.set_palette(palette)
plt.rcParams["figure.figsize"] = (8, 5)

# Paths
MODEL_PATHS = {
    'lightgbm': 'models/lgbm_regressor.pkl',
    'two_phase_classifier': 'models/two_phase_classifier.pkl',
    'two_phase_regressor': 'models/two_phase_regressor.pkl'
}
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
STATION_PATH = Path("data/raw/stations.csv")
OUTPUT_DIR = Path("figures/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Features selected based on mean absolute SHAP values from the LightGBM regressor
SELECTED_FEATURES = [
    'dep_same_hour_mean_7d', 'dep_lag_1', 'hour', 'dep_roll_mean_3', 'dep_lag_2',
    'station_id', 'temperature', 'dep_lag_24', 'dep_lag_168', 'is_weekend',
    'month', 'dep_roll_mean_24', 'weekday', 'precipitation', 'dep_roll_std_24',
    'year', 'dep_lag_72', 'dep_same_hour_std_7d', 'dep_lag_3', 'wind_speed'
]

def load_models():
    """
    Load trained LightGBM models from predefined paths.
    """
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"Warning: Model not found at {path}")
    return models

def plot_confusion_matrix(y_true, clf_preds):
    """
    Plot confusion matrix of the classification results.
    """
    cm = metrics.confusion_matrix(y_true, clf_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for LGBM Classifier')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png')
    plt.close()

def plot_shap_values(train_df):
    """
    Plot feature importance based on eman absolute shap values for 5000-row sample, 50 per station.
    """
    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df['departures']

    # Train the model
    reg = lgbm.LGBMRegressor(random_state=0, verbose=-1)
    reg.fit(X_train, y_train)

    # Sample 50 rows per station
    X_sample = (
        train_df.groupby("station_id", observed=True)[X_train.columns]
                .apply(lambda g: g.sample(n=min(50, len(g)), random_state=0))
                .reset_index(drop=True)
    )

    # Calculate the shap values
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X_sample)

    # Calculate mean absolute shap and sort by it
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "importance": shap_importance
    }).sort_values("importance", ascending=False)

    plt.figure()
    sns.barplot(x='importance', y='feature', data=shap_df, orient='h')
    plt.title('Feature importance based on mean absolute SHAP values')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_values.png')
    plt.close()

def plot_weekly_station_comparison(test_df, predictions, top_stations, station_indices=[0, 24, 79], start_day=105):
    """
    Plot observed vs. predicted bike demand over one week for stations
    with different usage levels.

    Args:
        test_df (pd.DataFrame): Test feature set including 'station_id'.
        predictions (dict): Dictionary of model predictions.
        top_stations (pd.Series): Ordered dataframe of top station ids and names by popularity.
        station_indices (list[int], optional): Indices of the stations (by popularity rank) to include in the plot.
        start_day (int, optional): Start day index in the test set. Defaults to 105.
    """
    def ordinal(n):
        """Returns the ordinal number string of n"""
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    
    line_styles = {
        "LightGBM": "-",
        "Two-Phase": "--",
        "Baseline": "-.",
    }

    y_true = test_df['departures']

    # Prepare station selection
    stations = top_stations.iloc[station_indices]
    station_labels = [
        f"{ordinal(i+1)} Most Popular Station - {station}" for i, station in zip(station_indices, stations['name'])
    ]

    # Get indices for selected station
    station_data = [
        test_df[test_df["station_id"] == station].index for station in stations['station_id']
    ]

    start_ind = 24 * start_day
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, ax in enumerate(axes):
        ind = station_data[i][start_ind: start_ind + 7 * 24]
        sns.lineplot(y_true[ind].to_numpy(), label="Observed", color="black", ax=ax)

        for name, preds in predictions.items():
            sns.lineplot(preds[ind], label=name, linestyle=line_styles.get(name, "-"), ax=ax)

        ax.set_title(f"{station_labels[i]} Station")
        ax.set_ylabel("Departures")
        ax.legend()

    fig.suptitle("Observed vs Predicted Hourly Departures for Selected Stations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'station_comparison.png')
    plt.close()

def plot_residuals(y_true, predictions, zoom_threshold=15):
    """
    Plot residuals vs. predicted values for multiple models using hexbin plots.

    Generates two rows of plots for each model:
        - Top row: full residual distribution
        - Bottom row: zoomed-in view showing residuals within ±zoom_threshold
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 9), sharey='row')

    for i, (name, preds) in enumerate(predictions.items()):
        residuals = y_true - preds

        # Full range plot
        hb = axes[0, i].hexbin(
            preds, residuals,
            gridsize=50,
            bins='log',
            mincnt=1
        )
        axes[0, i].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[0, i].set_title(f"{name}")
        if i == 0:
            axes[0, i].set_ylabel("Residuals")

        # Zoomed-in plot
        mask = np.abs(residuals) < zoom_threshold
        hb_zoom = axes[1, i].hexbin(
            preds[mask], residuals[mask],
            gridsize=50,
            bins='log',
            mincnt=1
        )
        axes[1, i].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, i].set_xlabel("Predicted Values")
        axes[1, i].set_title(f"{name} - Zoomed (±{zoom_threshold})")
        if i == 0:
            axes[1, i].set_ylabel("Residuals")

        last_hb = hb_zoom

    # Add shared colorbar
    if last_hb is not None:
        cb = fig.colorbar(last_hb, ax=axes.ravel().tolist())
        cb.set_label('log10(Count)')

    plt.suptitle("Residuals vs Predicted Values for Each Model (Hexbin)", fontsize=16)
    plt.savefig(OUTPUT_DIR / 'residuals.png')
    plt.close()

def main():
    """
    Main function to load data, compute predictions and generate evaluation plots.
    """
    # Load the data
    dtypes = {'hour': 'category', 'weekday': 'category', 'month': 'category', 'station_id': 'category'}
    try:
        train_df = load_csv(TRAIN_PATH, dtype=dtypes)
        test_df = load_csv(TEST_PATH, dtype=dtypes)
        station_df = load_csv(STATION_PATH, dtype={'id': 'category'})
    except FileNotFoundError as e:
        print(e)
        return
    
    # Check that selected features exist in the test dataframe
    for col in SELECTED_FEATURES:
        if col not in test_df.columns:
            print(f"Error: required feature '{col}' not found in test dataframe.")
            return
    
    target_col = 'departures'
    y_true = test_df[target_col]

    models = load_models()

    preds = {}

    # Evaluate the regression model
    if "lightgbm" in models:
        preds["LightGBM"] = models["lightgbm"].predict(test_df[SELECTED_FEATURES])
        print('LightGBM Regressor:')
        evaluate(y_true, preds["LightGBM"])

    # Evaluate the two-phase model
    if "two_phase_classifier" in models and "two_phase_regressor" in models:
        clf_preds = models["two_phase_classifier"].predict(test_df[SELECTED_FEATURES])
        reg_preds = models["two_phase_regressor"].predict(test_df[SELECTED_FEATURES])
        preds["Two-Phase"] = np.where(clf_preds == 0, 0, reg_preds)
        print('Two-Phase Model:')
        evaluate(y_true, preds["Two-Phase"])

        # Plot confusion matrix of the classifcation results
        plot_confusion_matrix((y_true > 0).astype(int), clf_preds)

    # Add naive baseline predictions (same hour one week prior)
    if 'dep_lag_168' in test_df.columns:
        preds["Baseline"] = test_df['dep_lag_168'].to_numpy()

    # Calculate station usage and sort
    top_station_ids = (
        pd.concat([train_df, test_df], axis=0).groupby('station_id', observed=True)[target_col]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    # Get names and ids of the top stations
    station_df.set_index('id', inplace=True)
    top_stations = station_df.loc[top_station_ids]['name'].reset_index(drop=False)

    plot_shap_values(train_df)
    plot_weekly_station_comparison(test_df, preds, top_stations)
    plot_residuals(y_true, preds)

    print(f"Evaluation plots saved: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()