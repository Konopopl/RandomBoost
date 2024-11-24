import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Literal


def is_importance_stable(fi, fi2, fi_perm, thresh, verbose=0):
    assert(len(fi) == len(fi2))

    best_importance_perm = np.max(fi_perm)
    is_feature_relevant = \
        (fi > best_importance_perm) & (fi2 > best_importance_perm)

    if np.all(~is_feature_relevant):
        return None, is_feature_relevant

    fi_diff = np.abs(fi-fi2)
    fi_max = np.array([fi, fi2]).max(axis=0)

    instability = np.mean(fi_diff/fi_max, where=is_feature_relevant)
    if verbose > 0:
        print('instability:', instability, end=' ')
    if verbose > 1:
        print('relevant features:', np.where(is_feature_relevant), end=' ')
    if verbose > 0:
        print('')

    is_stable = instability < thresh
    return is_stable, is_feature_relevant 


def get_adjusted_forest(
        random_forest_estimator,
        X, y,  
        min_n_estimators=None, 
        max_n_estimators=None,  
        thresh=0.1, verb=0):
    X, y = check_X_y(X, y)
    n, p = X.shape

    rf_params = random_forest_estimator.get_params()

    X_ = np.tile(X, (1, 3))
    if rf_params.get('random_state', None) is not None:
        np.random.seed(rf_params['random_state'])
    for j in range(p):
        np.random.shuffle(X_[:, j+2*p])

    if min_n_estimators is None:
        min_n_estimators = rf_params['n_estimators']
    if max_n_estimators is None:
        max_n_estimators = rf_params['n_estimators']

    random_forest_estimator.set_params(n_estimators=min_n_estimators)
    random_forest_estimator.fit(X_, y)
    if verb > 0:
        print('n_trees:', len(random_forest_estimator.estimators_), end=" ")

    is_stable, is_relevant = is_importance_stable(
        random_forest_estimator.feature_importances_[:p], 
        random_forest_estimator.feature_importances_[p:2*p], 
        random_forest_estimator.feature_importances_[2*p:], thresh, verb) 

    if np.all(~is_relevant):
        return None, np.where(is_relevant)
    if is_stable:
        return random_forest_estimator.fit(X[:, is_relevant], y), np.where(is_relevant)

    for n_estimators in range(min_n_estimators+1, max_n_estimators):
        random_forest_estimator.set_params(n_estimators=n_estimators, warm_start=True)
        random_forest_estimator.fit(X_, y)  # train 1 additional tree
        if verb > 0:
            print('n_trees:', len(random_forest_estimator.estimators_), end=" ")

        is_stable, is_relevant = is_importance_stable(
            random_forest_estimator.feature_importances_[:p], 
            random_forest_estimator.feature_importances_[p:2*p], 
            random_forest_estimator.feature_importances_[2*p:], thresh, verb)

        if np.all(~is_relevant):
            return None, np.where(is_relevant)
        if is_stable:
            random_forest_estimator.set_params(warm_start=False)
            return random_forest_estimator.fit(X[:, is_relevant], y), np.where(is_relevant)

    random_forest_estimator.set_params(warm_start=False)
    return random_forest_estimator.fit(X[:, is_relevant], y), np.where(is_relevant)
