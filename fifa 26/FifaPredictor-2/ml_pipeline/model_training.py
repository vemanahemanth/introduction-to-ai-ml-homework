import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve
)
import joblib
import logging
from typing import Dict, Tuple, List
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMTrainer:
    """Train and evaluate LightGBM model for FIFA finalist prediction."""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        
        self.params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'n_jobs': -1
        }
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series = None) -> Tuple:
        """Prepare data for training."""
        self.feature_names = list(X.columns)
        
        X_filled = X.fillna(X.median())
        
        for col in X_filled.columns:
            if X[col].isna().sum() > 0:
                X_filled[f'is_missing_{col}'] = X[col].isna().astype(int)
        
        logger.info(f"Prepared data: {X_filled.shape[0]} samples, {X_filled.shape[1]} features")
        return X_filled, y, groups
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, 
              y_val: pd.Series = None, num_boost_round: int = 500) -> lgb.Booster:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            num_boost_round: Maximum number of boosting rounds
            
        Returns:
            Trained LightGBM booster
        """
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        logger.info("Training LightGBM model...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        return self.model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_folds: int = 5) -> Dict:
        """
        Perform group k-fold cross-validation.
        
        Args:
            X: Features
            y: Labels
            groups: Group identifiers (e.g., tournament year)
            n_folds: Number of folds
            
        Returns:
            Dictionary containing CV metrics
        """
        gkf = GroupKFold(n_splits=n_folds)
        
        cv_scores = {
            'roc_auc': [],
            'pr_auc': [],
            'brier': [],
            'log_loss': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)
            
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            y_pred_proba = model.predict(X_val)
            
            cv_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            cv_scores['pr_auc'].append(average_precision_score(y_val, y_pred_proba))
            cv_scores['brier'].append(brier_score_loss(y_val, y_pred_proba))
            cv_scores['log_loss'].append(log_loss(y_val, y_pred_proba))
        
        cv_results = {
            'roc_auc_mean': np.mean(cv_scores['roc_auc']),
            'roc_auc_std': np.std(cv_scores['roc_auc']),
            'pr_auc_mean': np.mean(cv_scores['pr_auc']),
            'pr_auc_std': np.std(cv_scores['pr_auc']),
            'brier_mean': np.mean(cv_scores['brier']),
            'brier_std': np.std(cv_scores['brier']),
            'log_loss_mean': np.mean(cv_scores['log_loss']),
            'log_loss_std': np.std(cv_scores['log_loss']),
            'n_folds': n_folds
        }
        
        logger.info(f"CV Results: ROC-AUC = {cv_results['roc_auc_mean']:.3f} ± {cv_results['roc_auc_std']:.3f}")
        return cv_results
    
    def calibrate_model(self, X_cal: pd.DataFrame, y_cal: pd.Series, method: str = 'isotonic') -> CalibratedClassifierCV:
        """
        Calibrate model probabilities.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Calibrated model
        """
        logger.info(f"Calibrating model using {method} regression...")
        
        class LGBMWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict_proba(self, X):
                preds = self.model.predict(X)
                return np.vstack([1 - preds, preds]).T
        
        wrapped_model = LGBMWrapper(self.model)
        
        self.calibrated_model = CalibratedClassifierCV(
            wrapped_model,
            method=method,
            cv='prefit'
        )
        
        self.calibrated_model.fit(X_cal, y_cal)
        logger.info("Calibration completed")
        
        return self.calibrated_model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary containing all metrics
        """
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'threshold': threshold,
            'n_samples': len(y_test),
            'n_positive': int(y_test.sum()),
            'n_negative': int((1 - y_test).sum())
        }
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        
        self.metrics = metrics
        logger.info(f"Evaluation complete: ROC-AUC={metrics['roc_auc']:.3f}, Brier={metrics['brier_score']:.3f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_model(self, model_name: str = "lightgbm_model") -> None:
        """Save model and associated artifacts."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        model_path = self.output_dir / f"{model_name}.txt"
        self.model.save_model(str(model_path))
        
        if self.calibrated_model is not None:
            calibrated_path = self.output_dir / f"{model_name}_calibrated.pkl"
            joblib.dump(self.calibrated_model, calibrated_path)
        
        metrics_path = self.output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        feature_importance = self.get_feature_importance()
        importance_path = self.output_dir / f"{model_name}_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def load_model(self, model_name: str = "lightgbm_model") -> None:
        """Load saved model."""
        model_path = self.output_dir / f"{model_name}.txt"
        if model_path.exists():
            self.model = lgb.Booster(model_file=str(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
