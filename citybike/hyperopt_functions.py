import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK

def objective_reg(params, X, y, splits, two_phase=False):
    """
    Hyperopt objective function for a LightGBM regression model.

    Parameters:
        params (dict): Dictionary of the parameter space for LightGBM.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        splits (list of (np.ndarray, np.ndarray)): Cross-validation train/validation index splits.
        two_phase (bool): If True, only non-zero targets are used for training and evaluation.

    Returns:
        dict: Contains mean MAE ('loss') and Hyperopt status.
    """
    # Ensure integer hyperparameters
    params["num_leaves"] = int(params["num_leaves"])
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_child_samples"] = int(params["min_child_samples"])
    
    reg = LGBMRegressor(**params)
    scores = []
    for train_idx, val_idx in splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # If two_phase is True, only train and evaluate on non-zero targets
        if two_phase:
            pos_ind = y_train > 0
            reg.fit(X_train[pos_ind], y_train[pos_ind])
            preds = reg.predict(X_val)
            # Only compute MAE for non-zero validation targets
            scores.append(metrics.mean_absolute_error(y_val[y_val > 0], preds[y_val > 0]))
        else:
            reg.fit(X_train, y_train)
            preds = reg.predict(X_val)
            scores.append(metrics.mean_absolute_error(y_val, preds))
            
    return {'loss': np.mean(scores), 'status': STATUS_OK}

def objective_clf(params, X, y, splits):
    """
    Hyperopt objective function for a LightGBM classification model.

    Parameters:
        params (dict): Dictionary of the parameter space for LightGBM.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        splits (list of (np.ndarray, np.ndarray)): Cross-validation train/validation index splits.

    Returns:
        dict: Contains negative mean F1-score ('loss') and Hyperopt status.
    """
    # Ensure integer hyperparameters
    params["num_leaves"] = int(params["num_leaves"])
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_child_samples"] = int(params["min_child_samples"])
    
    clf = LGBMClassifier(**params)
    cv_scores = cross_val_score(clf, X, y, cv=splits, scoring='f1')

    # Negate F1-score because Hyperopt minimizes the loss
    return {'loss': -cv_scores.mean(), 'status': STATUS_OK}
