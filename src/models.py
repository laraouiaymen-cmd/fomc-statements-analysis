"""
Machine learning model wrappers for FOMC statement analysis.

This module provides clean wrapper classes for various ML models used to analyze
relationships between FOMC statement embeddings and S&P 500 movements. All models
accept pre-computed embeddings (from TF-IDF, BERT, or FinBERT) and provide
consistent interfaces.

CLASSIFICATION MODELS (for binary up/down label):
    - LogisticRegression
    - RandomForestClassifier
    - XGBoostClassifier

REGRESSION MODELS (for continuous % return):
    - ElasticNetRegressor
    - RandomForestRegressor
    - XGBoostRegressor
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.linear_model import ElasticNet as SKElasticNet
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from xgboost import XGBClassifier as XGBClassifierBase
from xgboost import XGBRegressor as XGBRegressorBase


# ============================================================================
# CLASSIFICATION MODELS - For predicting binary up/down direction (label)
# ============================================================================


class LogisticRegression:
    """
    Logistic Regression classifier wrapper.
    
    Args:
        penalty: Regularization type ("l2" or "elasticnet")
        C: Inverse of regularization strength (smaller = stronger)
        l1_ratio: ElasticNet mixing parameter (only used if penalty='elasticnet')
        max_iter: Maximum iterations for solver
        **kwargs: Additional parameters passed to sklearn LogisticRegression
    """
    
    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        **kwargs
    ):
        # Store init params for recreation
        self._init_params = {
            'penalty': penalty,
            'C': C,
            'l1_ratio': l1_ratio,
            'max_iter': max_iter,
            **kwargs
        }
        
        solver = 'saga' if penalty == 'elasticnet' else 'lbfgs'
        self.model = SKLogisticRegression(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
            solver=solver,
            max_iter=max_iter,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)


# ============================================================================
# REGRESSION MODELS - For predicting continuous % return (next_day_return)
# ============================================================================

class ElasticNetRegressor:
    """
    ElasticNet regression wrapper.
    
    Args:
        alpha: Regularization strength
        l1_ratio: ElasticNet mixing (0=Ridge, 1=Lasso)
        max_iter: Maximum iterations
        **kwargs: Additional parameters passed to sklearn ElasticNet
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        **kwargs
    ):
        self._init_params = {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'max_iter': max_iter,
            **kwargs
        }
        self.model = SKElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        return self.model.predict(X)


class RandomForestClassifier:
    """
    Random Forest classifier wrapper.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples to split node
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters passed to sklearn RandomForestClassifier
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        self._init_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state,
            **kwargs
        }
        self.model = SKRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class RandomForestRegressor:
    """
    Random Forest regressor wrapper.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples to split node
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters passed to sklearn RandomForestRegressor
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        self._init_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state,
            **kwargs
        }
        self.model = SKRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        return self.model.predict(X)


class XGBoostClassifier:
    """
    XGBoost classifier wrapper.
    
    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters passed to XGBClassifier
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        self._init_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            **kwargs
        }
        self.model = XGBClassifierBase(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class XGBoostRegressor:
    """
    XGBoost regressor wrapper.
    
    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters passed to XGBRegressor
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        self._init_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            **kwargs
        }
        self.model = XGBRegressorBase(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        return self.model.predict(X)


# ============================================================================
# Model Persistence Utilities
# ============================================================================

def save_model(model: Any, path: Path | str) -> None:
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: Trained model instance (any of the wrapper classes above)
        path: File path where model will be saved
        
    Example:
        save_model(trained_model, "models/logistic_tfidf.pkl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> Any:
    """
    Load a trained model from disk using joblib.
    
    Args:
        path: File path to the saved model
        
    Returns:
        Loaded model instance
        
    Example:
        model = load_model("models/logistic_tfidf.pkl")
    """
    return joblib.load(path)
