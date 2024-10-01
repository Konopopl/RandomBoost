from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble._forest import ForestClassifier  
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
import numpy as np

class BaseGradientBoostingClassifier(GradientBoostingClassifier):
    """
    Обертка над GradientBoostingClassifier для совместимости с ForestClassifier.
    """
    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0, 
        criterion = 'friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample, 
            criterion = criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
    def fit(self, X, y, sample_weight=None, check_input=True):
        # Игнорируем параметр check_input для совместимости
        #y = np.asarray(y)
        #if y.ndim > 1:
        #    y = y.ravel()  # Преобразуем y в одномерный массив
        return super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        # Игнорируем параметр check_input для совместимости
        return super().predict_proba(X)

    def predict(self, X, check_input=True):
        # Игнорируем параметр check_input для совместимости
        return super().predict(X)


class RandomBoostingClassifier(ForestClassifier):

    def __init__(
        self,
        n_estimators=100,
        *,
        gb_n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        base_estimator = BaseGradientBoostingClassifier(
            n_estimators=gb_n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
        )
        estimator_params = (
            'n_estimators',
            'learning_rate',
            'max_depth',
            'max_features',
            'random_state',
            'warm_start',
        )
        super().__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,  # Количество базовых моделей
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.gb_n_estimators = gb_n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples

        self.criterion = 'friedman_mse'  # Устанавливаем атрибут criterion для совместимости

    def _make_estimator(self, append=True, random_state=None):
        estimator = clone(self.estimator)
        estimator.set_params(
            n_estimators=self.gb_n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=random_state,
            warm_start=self.warm_start,
        )
        if append:
            self.estimators_.append(estimator)
        return estimator
    
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        all_importances = np.array(
            [est.feature_importances_ for est in self.estimators_]
        )
        mean_importances = np.mean(all_importances, axis=0)
        return mean_importances

    @property
    def feature_importances_var_(self):
        check_is_fitted(self)
        all_importances = np.array(
            [est.feature_importances_ for est in self.estimators_]
        )
        variances = np.var(all_importances, axis=0)
        return variances
