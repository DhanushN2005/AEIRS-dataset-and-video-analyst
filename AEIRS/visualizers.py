
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List

class Visualizer:
    def __init__(self, theme='plotly_white'):
        self.theme = theme
        self.colors = px.colors.qualitative.Set3
    
    def create_time_series_with_anomalies(self, df: pd.DataFrame, 
                                         date_col: str, 
                                         value_col: str, 
                                         anomalies: List[int]) -> str:
        """Create interactive time series plot with highlighted anomalies"""
        fig = go.Figure()
        
        # Main time series
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode='lines+markers',
            name='Values',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ))
        
        # Anomaly points
        if anomalies:
            anomaly_df = df.iloc[anomalies]
            fig.add_trace(go.Scatter(
                x=anomaly_df[date_col],
                y=anomaly_df[value_col],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#ef4444', size=12, symbol='x')
            ))
        
        fig.update_layout(
            title=f'{value_col} Over Time with Anomalies',
            xaxis_title=date_col,
            yaxis_title=value_col,
            template=self.theme,
            hovermode='x unified'
        )
        
        return fig.to_html()
    
    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            template=self.theme,
            height=600,
            width=800
        )
        
        return fig.to_html()
    
    def create_correlation_network(self, correlations: List[Dict]) -> str:
        """Create network graph of correlations"""
        import networkx as nx
        
        G = nx.Graph()
        
        for corr in correlations:
            G.add_edge(
                corr['feature1'], 
                corr['feature2'], 
                weight=abs(corr['pearson_correlation'])
            )
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edges
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*5, color='#ccc'),
                hoverinfo='none'
            ))
        
        # Create nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=20, color='#667eea'),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='Correlation Network',
            showlegend=False,
            template=self.theme,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig.to_html()
    
    def create_feature_importance_chart(self, features: List[str], 
                                       importances: List[float]) -> str:
        """Create feature importance bar chart"""
        fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template=self.theme,
            height=max(400, len(features) * 25)
        )
        
        return fig.to_html()
    
    def create_distribution_analysis(self, df: pd.DataFrame) -> str:
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=numeric_cols.tolist()
        )
        
        for idx, col in enumerate(numeric_cols, 1):
            row = (idx - 1) // 3 + 1
            col_pos = (idx - 1) % 3 + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=30),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title='Distribution Analysis',
            template=self.theme,
            height=600,
            showlegend=False
        )
        
        return fig.to_html()
    
    def create_sdg_radar_chart(self, sdg_scores: Dict) -> str:
        """Create radar chart for SDG scores"""
        categories = list(sdg_scores.keys())
        values = [sdg_scores[cat]['relevance_score'] for cat in categories]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            marker=dict(color='#667eea')
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='SDG Relevance Analysis',
            template=self.theme
        )
        
        return fig.to_html()
    
    def create_geographic_map(self, df: pd.DataFrame, 
                            lat_col: str, 
                            lon_col: str, 
                            value_col: str = None) -> str:
        """Create interactive map visualization"""
        if value_col:
            fig = px.scatter_mapbox(
                df, 
                lat=lat_col, 
                lon=lon_col,
                color=value_col,
                size=value_col,
                hover_data=df.columns,
                zoom=3
            )
        else:
            fig = px.scatter_mapbox(
                df, 
                lat=lat_col, 
                lon=lon_col,
                hover_data=df.columns,
                zoom=3
            )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            title='Geographic Distribution',
            template=self.theme
        )
        
        return fig.to_html()

