import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class Visualizer:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.risk_colors = {
            'Low': '#28a745',
            'Medium': '#ffc107',
            'High': '#fd7e14',
            'Critical': '#dc3545'
        }
    
    def create_rainfall_trend_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create rainfall trend chart"""
        try:
            fig = go.Figure()
            
            # Add rainfall line
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['rainfall'],
                mode='lines+markers',
                name='Rainfall (mm)',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=8)
            ))
            
            # Add threshold line
            threshold = data['rainfall'].mean() + 2 * data['rainfall'].std()
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Flood Risk Threshold"
            )
            
            fig.update_layout(
                title="7-Day Rainfall Trend",
                xaxis_title="Date",
                yaxis_title="Rainfall (mm)",
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating rainfall chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_river_level_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create river level monitoring chart"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=('River Level', 'Flow Rate'),
                vertical_spacing=0.1
            )
            
            # River level
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['level'],
                    mode='lines',
                    name='Water Level',
                    line=dict(color=self.colors['info'], width=2)
                ),
                row=1, col=1
            )
            
            # Add danger level
            danger_level = data['level'].quantile(0.8)
            fig.add_hline(
                y=danger_level,
                line_dash="dash",
                line_color="red",
                annotation_text="Danger Level"
            )
            
            # Flow rate (if available)
            if 'flow_rate' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['flow_rate'],
                        mode='lines',
                        name='Flow Rate',
                        line=dict(color=self.colors['secondary'], width=2)
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title="River Level Monitoring",
                height=400,
                hovermode='x unified'
            )
            
            fig.update_yaxes(title_text="Level (m)", row=1, col=1)
            fig.update_yaxes(title_text="Flow (m³/s)", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating river level chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_seasonal_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Create seasonal rainfall pattern analysis"""
        try:
            # Extract month from date
            data['month'] = pd.to_datetime(data['date']).dt.month
            
            # Calculate monthly averages
            monthly_avg = data.groupby('month')['rainfall'].mean().reset_index()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: months[x-1])
            
            fig = go.Figure()
            
            # Add bar chart for monthly averages
            fig.add_trace(go.Bar(
                x=monthly_avg['month_name'],
                y=monthly_avg['rainfall'],
                name='Average Rainfall',
                marker_color=self.colors['primary']
            ))
            
            # Add monsoon season highlighting
            monsoon_months = ['Jun', 'Jul', 'Aug', 'Sep']
            for month in monsoon_months:
                fig.add_vrect(
                    x0=months.index(month) - 0.4,
                    x1=months.index(month) + 0.4,
                    fillcolor="lightblue",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            
            fig.update_layout(
                title="Seasonal Rainfall Patterns",
                xaxis_title="Month",
                yaxis_title="Average Rainfall (mm)",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating seasonal analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_flood_events_timeline(self, data: pd.DataFrame) -> go.Figure:
        """Create timeline of historical flood events"""
        try:
            fig = go.Figure()
            
            # Create scatter plot for flood events
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['rainfall'],
                mode='markers',
                name='Flood Events',
                marker=dict(
                    size=data['severity'] * 5,  # Size based on severity
                    color=data['severity'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Severity"),
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=data.apply(lambda row: f"Date: {row['date']}<br>Rainfall: {row['rainfall']}mm<br>Severity: {row['severity']}", axis=1),
                hovertemplate="%{text}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Historical Flood Events Timeline",
                xaxis_title="Date",
                yaxis_title="Rainfall (mm)",
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating timeline: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_confusion_matrix(self, cm: List[List[int]]) -> go.Figure:
        """Create confusion matrix heatmap"""
        try:
            labels = ['No Flood', 'Flood']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title="Model Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                width=400,
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating confusion matrix: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_feature_importance_chart(self, importance: List[float], features: List[str]) -> go.Figure:
        """Create feature importance chart"""
        try:
            # Create DataFrame and sort
            df = pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=df['importance'],
                y=df['feature'],
                orientation='h',
                marker_color=self.colors['success']
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating feature importance chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_roc_curve(self, roc_data: Dict[str, Any]) -> go.Figure:
        """Create ROC curve"""
        try:
            fig = go.Figure()
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f'ROC Curve (AUC = {roc_data["auc"]:.3f})',
                line=dict(color=self.colors['primary'], width=3)
            ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=500,
                height=500
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating ROC curve: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
