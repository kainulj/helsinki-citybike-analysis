from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np

def cv(model, X, y, splits):
    """
    Cross-validates a regression model and prints MAE, RMSE, and R2 metrics.

    Parameters:
        model: The regression model to be cross-validated.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        splits (list of (np.ndarray, np.ndarray)): Cross-validation train/validation index splits.
    """
    cv_score = cross_validate(model, X, y, cv=splits, 
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    abs_error = -cv_score["test_neg_mean_absolute_error"]
    rmse = -cv_score["test_neg_root_mean_squared_error"]
    r2 = cv_score["test_r2"]

    print(f'Mean absolute error: {abs_error.mean():.3f} +/- {abs_error.std():.3f}')
    print(f'RMSE: {rmse.mean():.3f} +/- {rmse.std():.3f}')
    print(f'R2: {r2.mean():.3f} +/- {r2.std():.3f}')

def cv_two_phase(X, y, splits, clf, reg):
    """
    Cross-validates a two-phase model (classifier + regressor) and prints MAE, RMSE, and R2 metrics.

    In the two-phase approach, the classifier predicts whether the target is zero or non-zero,
    and the regressor predicts the value for non-zero cases.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        splits (list of (np.ndarray, np.ndarray)): Cross-validation train/validation index splits.
        clf: Classification model.
        reg: Regression model.
    """
    scores = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    for train_idx, val_idx in splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train the classifier to predict zero vs. non-zero targets
        clf.fit(X_train, (y_train > 0).astype(int))
        clf_preds = clf.predict(X_val)

        # Train the regressor only on non-zero targets
        pos_ind = y_train > 0
        reg.fit(X_train[pos_ind], y_train[pos_ind])
        reg_preds = reg.predict(X_val)
        
        # Set predictions to zero where classifier predicts zero
        reg_preds = np.where(clf_preds == 0, 0, reg_preds)

        scores['mae'].append(metrics.mean_absolute_error(y_val, reg_preds))
        scores['rmse'].append(metrics.root_mean_squared_error(y_val, reg_preds))
        scores['r2'].append(metrics.r2_score(y_val, reg_preds))

    print(f'Mean absolute error: {np.mean(scores["mae"]):.3f} +/- {np.std(scores["mae"]):.3f}')
    print(f'RMSE: {np.mean(scores["rmse"]):.3f} +/- {np.std(scores["rmse"]):.3f}')
    print(f'R2: {np.mean(scores["r2"]):.3f} +/- {np.std(scores["r2"]):.3f}')

def evaluate(y_true, y_preds):
    """
    Evaluates predictions and prints MAE, RMSE, and R2 metrics.

    Parameters:
        y_true (pd.Series): True target values.
        y_preds (np.array): Predicted values.
    """
    y_true = y_true.to_numpy()
    print(f'Mean absolute error: {metrics.mean_absolute_error(y_true, y_preds):.2f}')
    print(f'RMSE: {metrics.root_mean_squared_error(y_true, y_preds):.2f}')
    print(f'R2: {metrics.r2_score(y_true, y_preds):.3f}')