
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.detector = AnomalyDetector(
            threshold=config.get('anomaly_threshold', 3.0),
            contamination=config.get('contamination', 0.1)
        )
        self.corr_analyzer = CorrelationAnalyzer(
            threshold=config.get('correlation_threshold', 0.7)
        )
        self.causal_analyzer = CausalityAnalyzer()
        self.sdg_analyzer = SDGAnalyzer()
        self.visualizer = Visualizer()
        self.llm_explainer = LLMExplainer(
            provider=config.get('llm_provider', 'openai'),
            model=config.get('llm_model', 'gpt-4-turbo-preview')
        ) if config.get('enable_llm') else None
    
    async def run(self, df: pd.DataFrame, metadata: Dict) -> Dict:
        """Execute full analysis pipeline"""
        logger.info("Starting analysis pipeline...")
        
        results = {
            'metadata': metadata,
            'started_at': datetime.utcnow().isoformat(),
            'stages': {}
        }
        
        # Stage 1: Data Preprocessing
        logger.info("Stage 1: Preprocessing")
        df_clean = self._preprocess(df)
        results['stages']['preprocessing'] = {'status': 'completed', 'rows': len(df_clean)}
        
        # Stage 2: Anomaly Detection
        logger.info("Stage 2: Anomaly Detection")
        anomalies = self.detector.detect(df_clean)
        results['stages']['anomaly_detection'] = {
            'status': 'completed',
            'anomalies_found': sum(len(v['anomaly_indices']) for v in anomalies.values()),
            'details': anomalies
        }
        
        # Stage 3: Correlation Analysis
        logger.info("Stage 3: Correlation Analysis")
        correlations = self.corr_analyzer.analyze(df_clean)
        results['stages']['correlation_analysis'] = {
            'status': 'completed',
            'significant_pairs': len(correlations['significant_pairs']),
            'details': correlations
        }
        
        # Stage 4: Causality Analysis
        logger.info("Stage 4: Causality Analysis")
        causal_relationships = self.causal_analyzer.find_all_causal_relationships(df_clean)
        results['stages']['causality_analysis'] = {
            'status': 'completed',
            'causal_relationships': len(causal_relationships),
            'details': causal_relationships
        }
        
        # Stage 5: SDG Analysis
        if self.config.get('enable_sdg'):
            logger.info("Stage 5: SDG Analysis")
            sdg_results = self.sdg_analyzer.analyze(df_clean)
            results['stages']['sdg_analysis'] = {
                'status': 'completed',
                'sdgs_identified': len(sdg_results),
                'details': sdg_results
            }
        
        # Stage 6: Generate Visualizations
        logger.info("Stage 6: Generating Visualizations")
        visualizations = await self._generate_visualizations(df_clean, anomalies, correlations)
        results['stages']['visualizations'] = {
            'status': 'completed',
            'charts_generated': len(visualizations),
            'charts': visualizations
        }
        
        # Stage 7: LLM Explanations
        if self.llm_explainer:
            logger.info("Stage 7: Generating LLM Explanations")
            insights = await self._generate_insights(results)
            results['stages']['llm_insights'] = {
                'status': 'completed',
                'insights_generated': len(insights),
                'insights': insights
            }
        
        results['completed_at'] = datetime.utcnow().isoformat()
        results['status'] = 'completed'
        
        logger.info("Pipeline completed successfully")
        return results
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        df_clean = df.copy()
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Detect and parse dates
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                except:
                    pass
        
        return df_clean
    
    async def _generate_visualizations(self, df: pd.DataFrame, 
                                      anomalies: Dict, 
                                      correlations: Dict) -> List[Dict]:
        """Generate all visualizations"""
        charts = []
        
        # Time series with anomalies
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Limit to 3 time series
                if col in anomalies:
                    chart_html = self.visualizer.create_time_series_with_anomalies(
                        df, date_cols[0], col, anomalies[col]['anomaly_indices']
                    )
                    charts.append({
                        'name': f'Time Series: {col}',
                        'type': 'time_series',
                        'html': chart_html
                    })
        
        # Correlation heatmap
        if correlations['pearson_matrix']:
            corr_df = pd.DataFrame(correlations['pearson_matrix'])
            chart_html = self.visualizer.create_correlation_heatmap(corr_df)
            charts.append({
                'name': 'Correlation Heatmap',
                'type': 'heatmap',
                'html': chart_html
            })
        
        # Distribution analysis
        if len(numeric_cols) > 0:
            chart_html = self.visualizer.create_distribution_analysis(df)
            charts.append({
                'name': 'Distribution Analysis',
                'type': 'histogram',
                'html': chart_html
            })
        
        return charts
    
    async def _generate_insights(self, results: Dict) -> List[Dict]:
        """Generate LLM-powered insights"""
        insights = []
        
        # Anomaly insights
        anomalies = results['stages']['anomaly_detection']['details']
        for feature, data in list(anomalies.items())[:5]:  # Top 5
            if data['anomaly_indices']:
                explanation = self.llm_explainer.explain_anomaly({
                    'feature': feature,
                    'anomaly_count': len(data['anomaly_indices']),
                    'method': 'ensemble'
                })
                insights.append({
                    'type': 'anomaly',
                    'feature': feature,
                    'explanation': explanation,
                    'severity': 'high' if len(data['anomaly_indices']) > 10 else 'medium'
                })
        
        # Correlation insights
        correlations = results['stages']['correlation_analysis']['details']['significant_pairs']
        for corr in correlations[:5]:
            explanation = self.llm_explainer.explain_correlation(corr)
            insights.append({
                'type': 'correlation',
                'explanation': explanation,
                'severity': 'medium'
            })
        
        return insights


print("\n✅ Part 2/3 Generated:")
print("  - visualizers.py (Plotly charts)")
print("  - llm_explainer.py (LLM integration)")
print("  - pipeline.py (Core pipeline)")
print("\n📦 Ready for Part 3 (FastAPI server + deployment)...")