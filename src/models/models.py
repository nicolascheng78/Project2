"""
Machine learning models for stock prediction.
Implements classification (direction) and regression (return) models with calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)

import lightgbm as lgb
import xgboost as xgb


class StockClassifier:
    """Classification model for stock direction prediction with calibration."""
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict[str, Any]] = None,
        calibration_method: str = "isotonic"
    ):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost')
            params: Model hyperparameters
            calibration_method: Calibration method ('platt', 'isotonic')
        """
        self.model_type = model_type
        self.params = params or self._get_default_params()
        self.calibration_method = calibration_method
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the model type."""
        if self.model_type == "lightgbm":
            return {
                'objective': 'binary',
                'metric': 'auc',
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbose': -1
            }
        elif self.model_type == "xgboost":
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Fit the classification model.
        
        Args:
            X: Training features
            y: Training labels (0/1)
            X_val: Validation features
            y_val: Validation labels
        """
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train base model
        if self.model_type == "lightgbm":
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                self.model = lgb.LGBMClassifier(**self.params)
                self.model.fit(
                    X_scaled, y,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
                )
            else:
                self.model = lgb.LGBMClassifier(**self.params)
                self.model.fit(X_scaled, y)
                
        elif self.model_type == "xgboost":
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                self.model = xgb.XGBClassifier(**self.params)
                self.model.fit(
                    X_scaled, y,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=50,
                    verbose=100
                )
            else:
                self.model = xgb.XGBClassifier(**self.params)
                self.model.fit(X_scaled, y)
        
        # Calibrate probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=self.calibration_method,
            cv='prefit'
        )
        
        # Use validation set for calibration if available, otherwise use train set
        if X_val is not None and y_val is not None:
            X_cal_scaled = self.scaler.transform(X_val)
            self.calibrated_model.fit(X_cal_scaled, y_val)
        else:
            self.calibrated_model.fit(X_scaled, y)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of probabilities [P(down), P(up)]
        """
        X_scaled = self.scaler.transform(X)
        return self.calibrated_model.predict_proba(X_scaled)
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            threshold: Probability threshold for positive class
            
        Returns:
            Array of predictions (0/1)
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba),
            'brier': brier_score_loss(y, y_proba),
            'log_loss': log_loss(y, y_proba)
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the base model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'calibrated_model': self.calibrated_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'params': self.params,
                'calibration_method': self.calibration_method
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.calibrated_model = data['calibrated_model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.params = data['params']
        self.calibration_method = data['calibration_method']


class StockRegressor:
    """Regression model for stock return prediction with uncertainty estimation."""
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict[str, Any]] = None,
        quantile_regression: bool = True,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        """
        Initialize regressor.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost')
            params: Model hyperparameters
            quantile_regression: Whether to train quantile models
            quantiles: Quantiles to predict
        """
        self.model_type = model_type
        self.params = params or self._get_default_params()
        self.quantile_regression = quantile_regression
        self.quantiles = quantiles
        self.model = None
        self.quantile_models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the model type."""
        if self.model_type == "lightgbm":
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbose': -1
            }
        elif self.model_type == "xgboost":
            return {
                'objective': 'reg:squarederror',
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Fit the regression model.
        
        Args:
            X: Training features
            y: Training targets (returns)
            X_val: Validation features
            y_val: Validation targets
        """
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train main model
        if self.model_type == "lightgbm":
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                self.model = lgb.LGBMRegressor(**self.params)
                self.model.fit(
                    X_scaled, y,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
                )
            else:
                self.model = lgb.LGBMRegressor(**self.params)
                self.model.fit(X_scaled, y)
                
        elif self.model_type == "xgboost":
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                self.model = xgb.XGBRegressor(**self.params)
                self.model.fit(
                    X_scaled, y,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=50,
                    verbose=100
                )
            else:
                self.model = xgb.XGBRegressor(**self.params)
                self.model.fit(X_scaled, y)
        
        # Train quantile models for uncertainty estimation
        if self.quantile_regression and self.model_type == "lightgbm":
            for q in self.quantiles:
                q_params = self.params.copy()
                q_params['objective'] = 'quantile'
                q_params['alpha'] = q
                
                q_model = lgb.LGBMRegressor(**q_params)
                
                if X_val is not None and y_val is not None:
                    X_val_scaled = self.scaler.transform(X_val)
                    q_model.fit(
                        X_scaled, y,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                else:
                    q_model.fit(X_scaled, y)
                
                self.quantile_models[q] = q_model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict expected returns.
        
        Args:
            X: Features
            
        Returns:
            Array of predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Predict quantiles for uncertainty estimation.
        
        Args:
            X: Features
            
        Returns:
            Dictionary mapping quantile to predictions
        """
        if not self.quantile_models:
            raise ValueError("Quantile regression not enabled")
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for q, model in self.quantile_models.items():
            predictions[q] = model.predict(X_scaled)
        
        return predictions
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True values
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'quantile_models': self.quantile_models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'params': self.params,
                'quantile_regression': self.quantile_regression,
                'quantiles': self.quantiles
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.quantile_models = data['quantile_models']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.params = data['params']
        self.quantile_regression = data['quantile_regression']
        self.quantiles = data['quantiles']


class CombinedStockPredictor:
    """Combined classifier and regressor for comprehensive stock prediction."""
    
    def __init__(
        self,
        classifier_params: Optional[Dict[str, Any]] = None,
        regressor_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize combined predictor.
        
        Args:
            classifier_params: Parameters for classification model
            regressor_params: Parameters for regression model
        """
        self.classifier = StockClassifier(**(classifier_params or {}))
        self.regressor = StockRegressor(**(regressor_params or {}))
    
    def fit(
        self,
        X: pd.DataFrame,
        y_direction: pd.Series,
        y_return: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_direction_val: Optional[pd.Series] = None,
        y_return_val: Optional[pd.Series] = None
    ):
        """
        Fit both models.
        
        Args:
            X: Training features
            y_direction: Direction labels (0/1)
            y_return: Return targets
            X_val: Validation features
            y_direction_val: Validation direction labels
            y_return_val: Validation return targets
        """
        print("Training classification model...")
        self.classifier.fit(X, y_direction, X_val, y_direction_val)
        
        print("Training regression model...")
        self.regressor.fit(X, y_return, X_val, y_return_val)
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with both models.
        
        Args:
            X: Features
            
        Returns:
            Dictionary with predictions
        """
        return {
            'direction_proba': self.classifier.predict_proba(X)[:, 1],
            'direction': self.classifier.predict(X),
            'expected_return': self.regressor.predict(X),
            'return_quantiles': self.regressor.predict_quantiles(X) if self.regressor.quantile_models else None
        }
    
    def save(self, classifier_path: str, regressor_path: str):
        """Save both models."""
        self.classifier.save(classifier_path)
        self.regressor.save(regressor_path)
    
    def load(self, classifier_path: str, regressor_path: str):
        """Load both models."""
        self.classifier.load(classifier_path)
        self.regressor.load(regressor_path)
