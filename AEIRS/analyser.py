

# ============================================================================
# 6. analyzers.py - Correlation & Causality Analysis
# ============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Find significant correlations
        significant_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                corr_val = pearson_corr.iloc[i, j]
                
                if abs(corr_val) >= self.threshold:
                    # Calculate p-value
                    _, p_value = stats.pearsonr(numeric_df[col1].dropna(), 
                                                numeric_df[col2].dropna())
                    
                    significant_pairs.append({
                        'feature1': col1,
                        'feature2': col2,
                        'pearson_correlation': float(corr_val),
                        'spearman_correlation': float(spearman_corr.loc[col1, col2]),
                        'p_value': float(p_value),
                        'strength': self._interpret_correlation(abs(corr_val))
                    })
        
        return {
            'pearson_matrix': pearson_corr.to_dict(),
            'spearman_matrix': spearman_corr.to_dict(),
            'significant_pairs': significant_pairs
        }
    
    def _interpret_correlation(self, corr: float) -> str:
        if corr >= 0.9:
            return 'very_strong'
        elif corr >= 0.7:
            return 'strong'
        elif corr >= 0.5:
            return 'moderate'
        elif corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'

class CausalityAnalyzer:
    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag
    
    def granger_causality_test(self, df: pd.DataFrame, target_col: str, 
                               predictor_col: str) -> Dict:
   
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            data = df[[target_col, predictor_col]].dropna()
            
            if len(data) < 2 * self.max_lag:
                return {'error': 'Insufficient data for Granger test'}
            
            results = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
            
            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, self.max_lag + 1):
                p_value = results[lag][0]['ssr_ftest'][1]
                p_values[f'lag_{lag}'] = float(p_value)
            
            # Overall causality (if any lag is significant at 0.05)
            is_causal = any(p < 0.05 for p in p_values.values())
            
            return {
                'predictor': predictor_col,
                'target': target_col,
                'is_granger_causal': is_causal,
                'p_values_by_lag': p_values,
                'optimal_lag': min(p_values, key=p_values.get)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def find_all_causal_relationships(self, df: pd.DataFrame) -> List[Dict]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        causal_relationships = []
        
        for target in numeric_cols:
            for predictor in numeric_cols:
                if target != predictor:
                    result = self.granger_causality_test(df, target, predictor)
                    if result.get('is_granger_causal'):
                        causal_relationships.append(result)
        
        return causal_relationships

class SDGAnalyzer:
    def __init__(self):
        self.sdg_keywords = {
            1: ['poverty', 'poor', 'income', 'economic'],
            2: ['hunger', 'food', 'nutrition', 'agriculture'],
            3: ['health', 'medical', 'disease', 'mortality'],
            4: ['education', 'school', 'literacy', 'learning'],
            5: ['gender', 'women', 'equality', 'female'],
            6: ['water', 'sanitation', 'hygiene'],
            7: ['energy', 'electricity', 'renewable'],
            8: ['employment', 'work', 'economic growth', 'jobs'],
            9: ['infrastructure', 'innovation', 'industry'],
            10: ['inequality', 'discrimination', 'inclusion'],
            11: ['urban', 'city', 'housing', 'sustainable'],
            12: ['consumption', 'production', 'waste', 'recycling'],
            13: ['climate', 'emissions', 'carbon', 'greenhouse'],
            14: ['ocean', 'marine', 'sea', 'aquatic'],
            15: ['forest', 'biodiversity', 'land', 'ecosystem'],
            16: ['peace', 'justice', 'institutions', 'governance'],
            17: ['partnership', 'cooperation', 'global']
        }
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        detected_sdgs = {}
        
        # Analyze column names and text data
        all_text = ' '.join(df.columns.tolist()).lower()
        
        # Add string columns content
        for col in df.select_dtypes(include=['object']).columns:
            sample_text = ' '.join(df[col].dropna().astype(str).head(100).tolist())
            all_text += ' ' + sample_text.lower()
        
        for sdg_num, keywords in self.sdg_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in all_text)
            if matches > 0:
                detected_sdgs[f'SDG_{sdg_num}'] = {
                    'relevance_score': matches / len(keywords),
                    'matched_keywords': [kw for kw in keywords if kw in all_text]
                }
        
        return detected_sdgs

# Print first part
print("=" * 80)
print("AEIDS ENTERPRISE - COMPLETE PRODUCTION CODE")
print("=" * 80)
print("\n✅ Part 1/3 Generated:")
print("  - requirements.txt")
print("  - config.yaml")
print("  - models.py (Database models)")
print("  - auth.py (Authentication)")
print("  - detectors.py (Anomaly detection)")
print("  - analyzers.py (Correlation & Causality)")
print("\n📦 Continuing with Part 2...")
