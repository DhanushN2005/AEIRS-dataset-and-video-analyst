
# ============================================================================
# 5. detectors.py - Advanced Anomaly Detection
# ============================================================================
DETECTORS_PY = """
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, threshold: float = 3.0, contamination: float = 0.1):
        self.threshold = threshold
        self.contamination = contamination
        self.methods = {
            'zscore': self._zscore_detection,
            'isolation_forest': self._isolation_forest,
            'lof': self._lof_detection,
            'rolling_mad': self._rolling_mad,
            'iqr': self._iqr_detection
        }
    
    def detect(self, df: pd.DataFrame, methods: List[str] = None) -> Dict:
        if methods is None:
            methods = ['zscore', 'isolation_forest', 'rolling_mad']
        
        results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_results = {'anomalies': [], 'scores': []}
            
            for method in methods:
                if method in self.methods:
                    anomalies, scores = self.methods[method](df[col].values)
                    col_results['anomalies'].append(anomalies)
                    col_results['scores'].append(scores)
            
            # Ensemble voting
            if col_results['anomalies']:
                ensemble_anomalies = self._ensemble_vote(col_results['anomalies'])
                results[col] = {
                    'anomaly_indices': np.where(ensemble_anomalies)[0].tolist(),
                    'anomaly_values': df[col].iloc[ensemble_anomalies].tolist(),
                    'scores': np.mean(col_results['scores'], axis=0).tolist()
                }
        
        return results
    
    def _zscore_detection(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data = data[~np.isnan(data)]
        if len(data) < 3:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        z_scores = np.abs(stats.zscore(data))
        anomalies = z_scores > self.threshold
        return anomalies, z_scores
    
    def _isolation_forest(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data = data[~np.isnan(data)].reshape(-1, 1)
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        predictions = clf.fit_predict(data)
        scores = -clf.score_samples(data)
        anomalies = predictions == -1
        return anomalies, scores
    
    def _lof_detection(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data = data[~np.isnan(data)].reshape(-1, 1)
        if len(data) < 20:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        clf = LocalOutlierFactor(contamination=self.contamination, novelty=False)
        predictions = clf.fit_predict(data)
        scores = -clf.negative_outlier_factor_
        anomalies = predictions == -1
        return anomalies, scores
    
    def _rolling_mad(self, data: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < window:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        rolling_median = pd.Series(data).rolling(window=window, center=True).median()
        mad = np.abs(data - rolling_median)
        rolling_mad = pd.Series(mad).rolling(window=window, center=True).median()
        
        threshold = self.threshold * rolling_mad
        anomalies = mad > threshold
        scores = mad / (rolling_mad + 1e-10)
        
        return anomalies.fillna(False).values, scores.fillna(0).values
    
    def _iqr_detection(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q1, q3 = np.percentile(data[~np.isnan(data)], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = (data < lower_bound) | (data > upper_bound)
        scores = np.abs(data - np.median(data)) / (iqr + 1e-10)
        
        return anomalies, scores
    
    def _ensemble_vote(self, anomaly_lists: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        stacked = np.stack(anomaly_lists, axis=0)
        votes = np.mean(stacked, axis=0)
        return votes >= threshold

class TimeSeriesAnomalyDetector:
    def __init__(self):
        pass
    
    def detect_seasonal_anomalies(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
        \"\"\"Detect anomalies considering seasonality using Prophet-like decomposition\"\"\"
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            df = df.set_index(date_col)
            decomposition = seasonal_decompose(df[value_col], model='additive', period=7)
            
            residuals = decomposition.resid.dropna()
            threshold = 3 * residuals.std()
            
            anomalies = np.abs(residuals) > threshold
            
            return {
                'anomaly_indices': anomalies[anomalies].index.tolist(),
                'residuals': residuals.tolist(),
                'trend': decomposition.trend.dropna().tolist(),
                'seasonal': decomposition.seasonal.dropna().tolist()
            }
        except Exception as e:
            return {'error': str(e)}
"""