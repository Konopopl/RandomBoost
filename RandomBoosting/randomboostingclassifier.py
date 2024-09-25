from sklearn.ensemble._forest import ForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import (
    check_random_state, check_array, check_is_fitted, check_X_y
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import mode

@dataclass
class MyModel:
    model: GradientBoostingClassifier
    index: int
    random_state: int
    selected_features: np.ndarray  # Feature names
    selected_feature_indices: np.ndarray  # Feature indices
    validation_accuracy: float = 0.0
    feature_importance_dict: dict = field(default_factory=dict)
    training_time: float = 0.0

class RandomBoostingClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=60,  # Number of models in the ensemble
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
        gb_n_estimators=100,  # Number of trees in each GradientBoostingClassifier
    ):
        super().__init__(
            estimator=GradientBoostingClassifier(),
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
        self.max_features = max_features  # For base class
        self.gb_max_features = max_features  # For GradientBoostingClassifier
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

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def fit(self, X, y, sample_weight=None):
        # Save feature names before converting X
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(['X' + str(i) for i in range(X.shape[1])])

        X, y = check_X_y(X, y)

        y = self._validate_y_class_weight(y)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1  # Set number of outputs

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
            raise ValueError("Invalid value for model_features")

    def _initialize_models(self):
        self.models = []
        rng = check_random_state(self.random_state)
        for i in range(self.n_estimators):
            model_random_state = rng.randint(np.iinfo(np.int32).max)
            selected_feature_indices = rng.choice(
                self.n_features_in_, size=self.my_model_features, replace=False
            )
            selected_features = self.feature_names_in_[selected_feature_indices]
            model = GradientBoostingClassifier(
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
                selected_features=selected_features,  # Feature names
                selected_feature_indices=selected_feature_indices  # Indices
            )
            self.models.append(my_model)

    def _update_models(self):
        existing_n_estimators = len(self.models)
        rng = check_random_state(self.random_state)
        if self.n_estimators > existing_n_estimators:
            # Add new models
            for i in range(existing_n_estimators, self.n_estimators):
                model_random_state = rng.randint(np.iinfo(np.int32).max)
                selected_feature_indices = rng.choice(
                    self.n_features_in_, size=self.my_model_features, replace=False
                )
                selected_features = self.feature_names_in_[selected_feature_indices]
                model = GradientBoostingClassifier(
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
            # Remove excess models
            self.models = self.models[:self.n_estimators]

        # Update existing models
        for my_model in self.models:
            current_gb_n_estimators = my_model.model.get_params()['n_estimators']
            if self.warm_start:
                if self.gb_n_estimators > current_gb_n_estimators:
                    # Continue training by increasing n_estimators
                    my_model.model.set_params(
                        n_estimators=self.gb_n_estimators,
                        warm_start=True
                    )
                elif self.gb_n_estimators == current_gb_n_estimators:
                    # No changes needed
                    pass
                else:
                    # Cannot decrease n_estimators with warm_start, retrain model
                    my_model.model = GradientBoostingClassifier(
                        n_estimators=self.gb_n_estimators,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth,
                        max_features=self.gb_max_features,
                        subsample=self.subsample,
                        random_state=my_model.random_state,
                        warm_start=self.warm_start,
                    )
            else:
                # If warm_start=False, retrain model
                my_model.model = GradientBoostingClassifier(
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
        my_model.validation_accuracy = my_model.model.score(X_train, y)
        if hasattr(my_model.model, 'feature_importances_'):
            feature_importances = my_model.model.feature_importances_
            # Use feature names
            my_model.feature_importance_dict = dict(zip(my_model.selected_features, feature_importances))
        else:
            my_model.feature_importance_dict = {}
        return my_model

    def _update_models_trust(self):
        self.modelsTrast = [model for model in self.models if model.validation_accuracy >= self.level_of_trust]
        if not self.modelsTrast:
            self.modelsTrast = self.models

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        predictions = np.array([
            my_model.model.predict(X[:, my_model.selected_feature_indices])
            for my_model in self.modelsTrast
        ])
        return mode(predictions, axis=0, keepdims=False).mode.flatten()

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        proba_sum = sum(
            my_model.model.predict_proba(X[:, my_model.selected_feature_indices])
            for my_model in self.modelsTrast
        )
        avg_proba = proba_sum / len(self.modelsTrast)
        return avg_proba

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
        # Compute the variance of feature importance across all models in the ensemble.
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
        # Update base class parameters
        base_params = {key: params[key] for key in params if key in self.__class__.__base__.__init__.__code__.co_varnames}
        super().set_params(**base_params)
        return self
