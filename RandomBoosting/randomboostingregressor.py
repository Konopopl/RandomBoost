from sklearn.ensemble._forest import ForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import (
    check_random_state, check_array, check_is_fitted, check_X_y
)
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dataclasses import dataclass, field 
import time

@dataclass
class MyModel:
    model: GradientBoostingRegressor
    index: int
    random_state: int
    selected_features: np.ndarray  # Имена признаков
    selected_feature_indices: np.ndarray  # Индексы признаков
    validation_score: float = 0.0
    feature_importance_dict: dict = field(default_factory=dict)
    training_time: float = 0.0

class RandomBoostingRegressor(ForestRegressor):
    def __init__(
        self,
        n_estimators=60,  # Количество моделей в ансамбле
        learning_rate=0.1,
        max_depth=2,
        max_features=1.0,
        subsample=1.0,
        model_features=1.0,
        random_state=None,
        n_jobs=None,
        level_of_trust=0.0,
        voting_weights=None,
        warm_start=False,
        gb_n_estimators=100,  # Количество деревьев в каждой модели GradientBoostingRegressor
    ):
        super().__init__(
            estimator=GradientBoostingRegressor(),
            n_estimators=n_estimators,
            estimator_params=(),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
            max_samples=None,
        )
        self.gb_n_estimators = gb_n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features  # Для базового класса
        self.gb_max_features = max_features  # Для GradientBoostingRegressor
        self.subsample = subsample
        self.model_features = model_features
        self.level_of_trust = level_of_trust
        self.voting_weights = voting_weights
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.warm_start = warm_start

        self.models = []
        self.modelsTrast = []
        self.is_fitted_ = False

    def fit(self, X, y, sample_weight=None):
        # Сохраняем названия признаков до преобразования X
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(['X' + str(i) for i in range(X.shape[1])])

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1  # Устанавливаем количество выходов

        self._determine_model_features()

        if not self.is_fitted_:
            self._initialize_models()
            self.is_fitted_ = True
        else:
            self._update_models()

        self._train_models(X, y, sample_weight)
        self._update_models_trust()
        return self

    def _determine_model_features(self):
        if isinstance(self.model_features, int):
            self.my_model_features = max(1, min(self.model_features, self.n_features_in_))
        elif isinstance(self.model_features, float):
            self.my_model_features = max(1, int(self.model_features * self.n_features_in_))
            self.my_model_features = min(self.my_model_features, self.n_features_in_)
        elif self.model_features == 'sqrt':
            self.my_model_features = max(1, int(np.sqrt(self.n_features_in_)))
        else:
            raise ValueError("Некорректное значение для model_features")

    def _initialize_models(self):
        self.models = []
        rng = check_random_state(self.random_state)
        for i in range(self.n_estimators):
            model_random_state = rng.randint(np.iinfo(np.int32).max)
            selected_feature_indices = rng.choice(
                self.n_features_in_, size=self.my_model_features, replace=False
            )
            selected_features = self.feature_names_in_[selected_feature_indices]
            model = GradientBoostingRegressor(
                n_estimators=self.gb_n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_features=self.gb_max_features,
                subsample=self.subsample,
                random_state=model_random_state,
                warm_start=self.warm_start,
            )
            my_model = MyModel(
                model=model,
                index=i,
                random_state=model_random_state,
                selected_features=selected_features,  # Имена признаков
                selected_feature_indices=selected_feature_indices  # Индексы признаков
            )
            self.models.append(my_model)

    def _update_models(self):
        existing_n_estimators = len(self.models)
        rng = check_random_state(self.random_state)
        if self.n_estimators > existing_n_estimators:
            # Добавляем новые модели
            for i in range(existing_n_estimators, self.n_estimators):
                model_random_state = rng.randint(np.iinfo(np.int32).max)
                selected_feature_indices = rng.choice(
                    self.n_features_in_, size=self.my_model_features, replace=False
                )
                selected_features = self.feature_names_in_[selected_feature_indices]
                model = GradientBoostingRegressor(
                    n_estimators=self.gb_n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    max_features=self.gb_max_features,
                    subsample=self.subsample,
                    random_state=model_random_state,
                    warm_start=self.warm_start,
                )
                my_model = MyModel(
                    model=model,
                    index=i,
                    random_state=model_random_state,
                    selected_features=selected_features,
                    selected_feature_indices=selected_feature_indices
                )
                self.models.append(my_model)
        elif self.n_estimators < existing_n_estimators:
            # Удаляем лишние модели
            self.models = self.models[:self.n_estimators]

        # Обновляем существующие модели
        for my_model in self.models:
            current_gb_n_estimators = my_model.model.get_params()['n_estimators']
            if self.warm_start:
                if self.gb_n_estimators > current_gb_n_estimators:
                    # Продолжаем обучение, увеличивая n_estimators
                    my_model.model.set_params(
                        n_estimators=self.gb_n_estimators,
                        warm_start=True
                    )
                elif self.gb_n_estimators == current_gb_n_estimators:
                    # Никаких изменений не требуется
                    pass
                else:
                    # Не можем уменьшить n_estimators с warm_start, переобучаем модель
                    my_model.model = GradientBoostingRegressor(
                        n_estimators=self.gb_n_estimators,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth,
                        max_features=self.gb_max_features,
                        subsample=self.subsample,
                        random_state=my_model.random_state,
                        warm_start=self.warm_start,
                    )
            else:
                # Если warm_start=False, переобучаем модель
                my_model.model = GradientBoostingRegressor(
                    n_estimators=self.gb_n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    max_features=self.gb_max_features,
                    subsample=self.subsample,
                    random_state=my_model.random_state,
                    warm_start=self.warm_start,
                )

    def _train_models(self, X, y, sample_weight):
        parallel = Parallel(n_jobs=self.n_jobs)
        self.models = parallel(
            delayed(self._fit_model)(my_model, X, y, sample_weight)
            for my_model in self.models
        )

    def _fit_model(self, my_model, X, y, sample_weight):
        X_train = X[:, my_model.selected_feature_indices]
        my_model.model.fit(X_train, y, sample_weight=sample_weight)
        # Используем R^2 score для оценки модели
        my_model.validation_score = my_model.model.score(X_train, y)
        if hasattr(my_model.model, 'feature_importances_'):
            feature_importances = my_model.model.feature_importances_
            # Используем имена признаков
            my_model.feature_importance_dict = dict(zip(my_model.selected_features, feature_importances))
        else:
            my_model.feature_importance_dict = {}
        return my_model

    def _update_models_trust(self):
        # Отбираем модели, у которых validation_score >= level_of_trust
        self.modelsTrast = [model for model in self.models if model.validation_score >= self.level_of_trust]
        if not self.modelsTrast:
            self.modelsTrast = self.models

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        predictions = np.array([
            my_model.model.predict(X[:, my_model.selected_feature_indices])
            for my_model in self.modelsTrast
        ])
        # Используем среднее значение предсказаний
        return np.mean(predictions, axis=0)

    @property
    def feature_importances_(self):
        importances = pd.Series(0, index=self.feature_names_in_, dtype=float)
        for model in self.modelsTrast:
            importances_model = pd.Series(model.feature_importance_dict)
            importances = importances.add(importances_model, fill_value=0)
        importances /= len(self.modelsTrast)
        return importances.values

    @property
    def feature_importances_var_(self):
        # Вычисляем дисперсию важности признаков по всем моделям
        df_importance = pd.DataFrame(
            [model.feature_importance_dict for model in self.modelsTrast]
        ).fillna(0)
        variances = df_importance.var(skipna=True)
        return variances.reindex(self.feature_names_in_, fill_value=0).values

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({
            'n_estimators': self.n_estimators,
            'gb_n_estimators': self.gb_n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'subsample': self.subsample,
            'model_features': self.model_features,
            'level_of_trust': self.level_of_trust,
            'voting_weights': self.voting_weights,
            'warm_start': self.warm_start,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        })
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in ['n_estimators', 'gb_n_estimators', 'learning_rate', 'max_depth', 'max_features',
                       'subsample', 'model_features', 'level_of_trust', 'voting_weights', 'warm_start',
                       'random_state', 'n_jobs']:
                setattr(self, key, value)
        # Обновляем параметры базового класса
        base_params = {key: params[key] for key in params if key in self.__class__.__base__.__init__.__code__.co_varnames}
        super().set_params(**base_params)
        return self
