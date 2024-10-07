from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble._forest import ForestClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
import numpy as np

class BaseGradientBoostingClassifier(GradientBoostingClassifier):
    """
    Обертка над GradientBoostingClassifier для совместимости с ForestClassifier.
    """
    def fit(self, X, y, sample_weight=None, check_input=True):
        return super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        return super().predict_proba(X)

    def predict(self, X, check_input=True):
        return super().predict(X)

class RandomBoostingClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        *,
        gb_n_estimators=100,
        loss='log_loss',
        learning_rate=0.1,
        subsample=1.0,
        criterion='friedman_mse',
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
        # Параметры ForestClassifier
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        max_samples=None,
    ):
        # Создаем базовый оценщик с параметрами градиентного бустинга
        base_estimator = BaseGradientBoostingClassifier(
            n_estimators=gb_n_estimators,
            loss=loss,
            learning_rate=learning_rate,
            subsample=subsample,
            criterion=criterion,
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

        # Параметры, которые будут использоваться для настройки базовых оценщиков
        estimator_params = (
            'n_estimators',
            'loss',
            'learning_rate',
            'subsample',
            'criterion',
            'min_samples_split',
            'min_samples_leaf',
            'min_weight_fraction_leaf',
            'max_depth',
            'min_impurity_decrease',
            'init',
            'random_state',
            'max_features',
            'verbose',
            'max_leaf_nodes',
            'warm_start',
            'validation_fraction',
            'n_iter_no_change',
            'tol',
            'ccp_alpha',
        )

        # Инициализируем ForestClassifier
        super().__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        # Сохраняем все параметры как атрибуты
        self.gb_n_estimators = gb_n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.max_samples = max_samples

    def _make_estimator(self, append=True, random_state=None):
        estimator = clone(self.estimator_)
        # Собираем параметры для базового оценщика
        params = {param: getattr(self, param) for param in self.estimator_params}
        params['n_estimators'] = self.gb_n_estimators  # Устанавливаем gb_n_estimators
        estimator.set_params(**params)
        if random_state is not None:
            _set_random_states(estimator, random_state)
        if append:
            self.estimators_.append(estimator)
        return estimator

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        all_importances = np.array(
            [est.feature_importances_ for est in self.estimators_]
        )
        return np.mean(all_importances, axis=0)

    @property
    def feature_importances_var_(self):
        check_is_fitted(self)
        all_importances = np.array(
            [est.feature_importances_ for est in self.estimators_]
        )
        return np.var(all_importances, axis=0)
