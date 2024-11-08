from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed
from scipy.stats import mode
import time

@dataclass
class MyModel:
    model: GradientBoostingClassifier
    index: int
    random_state: int
    selected_features: np.ndarray
    validation_accuracy: float = 0.0
    feature_importance_dict: dict = field(default_factory=dict)
    training_time: float = 0.0

class RandomBoosting(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        max_features=1.0,
        subsample=1.0,
        n_models=60,
        model_features=1.0,  # Number of features for each model
        random_state=42,
        n_jobs=-1,
        level_of_trust=0.0,
        voting_weights=0.0,
        warm_start=False,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.subsample = subsample
        # Parameters for the ensemble
        self.n_models = n_models
        self.model_features = model_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.level_of_trust = level_of_trust
        self.voting_weights = voting_weights
        self.warm_start = warm_start
        self.kwargs = kwargs  # Additional parameters for the base model

        self.models = []
        self.modelsTrast = []
        self.is_fitted_ = False

    def fit(self, X, y):
        # Initialize necessary attributes
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f'feature_{i}' for i in range(self.n_features_in_)])
        self.classes_ = np.unique(y)

        # Store training data
        self.X_train = X.reset_index(drop=True)
        self.y_train = y.reset_index(drop=True)

        # Determine the number of features for each model
        self._determine_model_features()

        if not self.is_fitted_:
            # First fit: initialize models
            self._initialize_models()
            self.is_fitted_ = True
        else:
            # Update models for warm_start
            self._update_models()

        # Train models
        self._train_models()

        # Update trusted models based on level_of_trust
        self._update_models_trust()

        return self

    def _determine_model_features(self):
        if isinstance(self.model_features, int):
            self.my_model_features = max(1, self.model_features)
        elif isinstance(self.model_features, float):
            self.my_model_features = max(1, int(self.model_features * self.n_features_in_))
        elif self.model_features == 'sqrt':
            self.my_model_features = max(1, int(np.sqrt(self.n_features_in_)))
        else:
            raise ValueError("Invalid value for model_features")

    def _initialize_models(self):
        self.models = []
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_models):
            model_random_state = self.random_state + i
            # Select features for the model
            selected_features = rng.choice(
                self.feature_names_in_, size=self.my_model_features, replace=False
            )
            # Create the base model
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_features=self.max_features,
                subsample=self.subsample,
                random_state=model_random_state,
                warm_start=self.warm_start,
                **self.kwargs
            )
            my_model = MyModel(
                model=model,
                index=i,
                random_state=model_random_state,
                selected_features=selected_features,
            )
            self.models.append(my_model)

    def _update_models(self):
        for my_model in self.models:
            current_n_estimators = my_model.model.get_params()['n_estimators']
            if self.warm_start and self.n_estimators >= current_n_estimators:
                # Continue training with increased number of estimators
                my_model.model.set_params(
                    n_estimators=self.n_estimators,
                    warm_start=True
                )
            else:
                # Re-initialize the model
                model = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    subsample=self.subsample,
                    random_state=my_model.random_state,
                    warm_start=self.warm_start,
                    **self.kwargs
                )
                my_model.model = model
                # Selected features remain the same

    def _train_models(self):
        start_time = time.time()

        # Use parallel training if n_jobs != 1
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_model)(my_model)
            for my_model in self.models
        )

        self.max_time = time.time() - start_time

    def fit_model(self, my_model):
        X_train = self.X_train[my_model.selected_features]
        y_train = self.y_train

        # Train the model
        start_time = time.time()
        my_model.model.fit(X_train, y_train)
        my_model.training_time += time.time() - start_time

        # Evaluate validation accuracy (on training data here, but can be modified)
        my_model.validation_accuracy = my_model.model.score(X_train, y_train)

        # Compute feature importances
        if hasattr(my_model.model, 'feature_importances_'):
            feature_importances = my_model.model.feature_importances_
            my_model.feature_importance_dict = dict(zip(my_model.selected_features, feature_importances))
        else:
            my_model.feature_importance_dict = {}

        return my_model

    def _update_models_trust(self):
        self.modelsTrast = [model for model in self.models
                            if model.validation_accuracy >= self.level_of_trust]
        if not self.modelsTrast:
            # If no models meet the level_of_trust, use all models
            self.modelsTrast = self.models

    def predict(self, X):
        check_is_fitted(self, 'models')
        predictions = np.array([
            my_model.model.predict(X[my_model.selected_features])
            for my_model in self.modelsTrast
        ])
        # Majority voting
        return mode(predictions, axis=0, keepdims=False).mode.flatten()

    def predict_proba(self, X):
        check_is_fitted(self, 'models')
        proba_sum = sum(
            my_model.model.predict_proba(X[my_model.selected_features])
            for my_model in self.modelsTrast
        )
        avg_proba = proba_sum / len(self.modelsTrast)
        return avg_proba

    @property
    def feature_importances_(self):
        importances_list = [
            [model.feature_importance_dict.get(feature, 0.0) for feature in self.feature_names_in_]
            for model in self.modelsTrast
        ]
        means = np.mean(importances_list, axis=0)
        return means 
    @property
    def feature_importances_var_(self):
        # Compute the variance of feature importance across all models in the ensemble.
        df_importance = pd.DataFrame(
            [model.feature_importance_dict for model in self.modelsTrast],
            columns=self.feature_names_in_
        )
        variances = df_importance.var(skipna=True)
        return variances.values

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'subsample': self.subsample,
            'n_models': self.n_models,
            'model_features': self.model_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'level_of_trust': self.level_of_trust,
            'voting_weights': self.voting_weights,
            'warm_start': self.warm_start,
            **self.kwargs
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
