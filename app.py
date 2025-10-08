import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
from datetime import datetime, timedelta
import os

from models.flood_predictor import FloodPredictor
from data.data_loader import DataLoader
from utils.visualization import Visualizer
from utils.risk_assessment import RiskAssessment
from config.settings import INDIAN_STATES, RISK_COLORS, MODEL_FEATURES

# Page configuration
st.set_page_config(
    page_title="AI Flood Prediction System - India",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'risk_assessor' not in st.session_state:
    st.session_state.risk_assessor = RiskAssessment()

def main():
    st.title("🌊 AI-Powered Flood Prediction System for India")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard", "Real-time Prediction", "AI Insights", "Historical Analysis", "Model Performance", "Risk Mapping"]
    )
    
    # Load model if not already loaded
    if st.session_state.model is None:
        with st.spinner("Loading ML model..."):
            try:
                st.session_state.model = FloodPredictor()
                st.session_state.model.load_or_train_model()
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return
    
    # Route to different pages
    if page == "Dashboard":
        show_dashboard()
    elif page == "Real-time Prediction":
        show_real_time_prediction()
    elif page == "AI Insights":
        show_ai_insights()
    elif page == "Historical Analysis":
        show_historical_analysis()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Risk Mapping":
        show_risk_mapping()

def show_dashboard():
    st.header("📊 Flood Monitoring Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Weather Conditions")
        
        # State selection
        selected_state = st.selectbox("Select State:", list(INDIAN_STATES.keys()))
        
        # Load current weather data
        try:
            current_data = st.session_state.data_loader.get_current_weather_data(selected_state)
            if current_data is not None:
                # Display current conditions
                col1a, col1b, col1c = st.columns(3)
                
                with col1a:
                    st.metric("Rainfall (mm)", f"{current_data.get('rainfall', 0):.1f}")
                
                with col1b:
                    st.metric("River Level (m)", f"{current_data.get('river_level', 0):.2f}")
                
                with col1c:
                    risk_level = st.session_state.risk_assessor.assess_risk(
                        current_data.get('rainfall', 0),
                        current_data.get('river_level', 0),
                        current_data.get('temperature', 25)
                    )
                    st.metric("Risk Level", risk_level, delta_color="inverse")
                
                # Rainfall trend chart
                st.subheader("7-Day Rainfall Trend")
                rainfall_data = st.session_state.data_loader.get_rainfall_trend(selected_state, days=7)
                if rainfall_data is not None:
                    fig = st.session_state.visualizer.create_rainfall_trend_chart(rainfall_data)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Rainfall trend data not available")
            else:
                st.warning(f"Current weather data not available for {selected_state}")
        except Exception as e:
            st.error(f"Error loading current data: {str(e)}")
    
    with col2:
        st.subheader("River Level Monitoring")
        
        try:
            river_data = st.session_state.data_loader.get_river_level_data(selected_state)
            if river_data is not None:
                fig = st.session_state.visualizer.create_river_level_chart(river_data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"River level data not available for {selected_state}")
        except Exception as e:
            st.error(f"Error loading river data: {str(e)}")
    
    # National flood risk overview
    st.subheader("🗺️ National Flood Risk Overview")
    try:
        national_data = st.session_state.data_loader.get_national_flood_data()
        if national_data is not None:
            col3, col4, col5, col6 = st.columns(4)
            
            risk_counts = national_data['risk_level'].value_counts()
            
            with col3:
                st.metric("Low Risk Regions", risk_counts.get('Low', 0), delta_color="normal")
            with col4:
                st.metric("Medium Risk Regions", risk_counts.get('Medium', 0), delta_color="normal")
            with col5:
                st.metric("High Risk Regions", risk_counts.get('High', 0), delta_color="inverse")
            with col6:
                st.metric("Critical Risk Regions", risk_counts.get('Critical', 0), delta_color="inverse")
        else:
            st.info("National flood risk data will be displayed here when available from official sources.")
    except Exception as e:
        st.error(f"Error loading national data: {str(e)}")

def show_real_time_prediction():
    st.header("🔮 Real-time Flood Prediction")
    
    st.markdown("Enter current meteorological conditions to get flood risk prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input parameters
        st.subheader("Input Parameters")
        
        rainfall = st.slider("Current Rainfall (mm)", 0.0, 200.0, 10.0, 0.1)
        river_level = st.slider("River Level (m)", 0.0, 20.0, 5.0, 0.1)
        temperature = st.slider("Temperature (°C)", 15.0, 45.0, 25.0, 0.1)
        humidity = st.slider("Humidity (%)", 30.0, 100.0, 70.0, 1.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0, 1.0)
        
        # Additional parameters
        previous_rainfall = st.slider("Previous 24h Rainfall (mm)", 0.0, 300.0, 5.0, 0.1)
        soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 50.0, 1.0)
        elevation = st.slider("Elevation (m)", 0, 3000, 200, 10)
    
    with col2:
        st.subheader("Prediction Results")
        
        if st.button("Predict Flood Risk", type="primary"):
            try:
                # Prepare input data
                input_data = np.array([[
                    rainfall, river_level, temperature, humidity, wind_speed,
                    previous_rainfall, soil_moisture, elevation
                ]])
                
                # Make prediction
                prediction = st.session_state.model.predict(input_data)
                probability = st.session_state.model.predict_probability(input_data)
                
                # Display results
                risk_level = st.session_state.risk_assessor.get_risk_level_from_probability(probability[0])
                
                # Risk level display
                risk_color = RISK_COLORS.get(risk_level, "#808080")
                st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {risk_color}; color: white; text-align: center; margin: 20px 0;">
                        <h2>Risk Level: {risk_level}</h2>
                        <h3>Flood Probability: {probability[0]:.1%}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.subheader("Risk Breakdown")
                prob_data = {
                    'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
                    'Probability': [
                        max(0, 0.6 - probability[0]) if probability[0] < 0.6 else 0,
                        max(0, min(0.3, 0.8 - probability[0])) if 0.4 <= probability[0] <= 0.8 else 0,
                        max(0, min(0.2, probability[0] - 0.6)) if probability[0] > 0.6 else 0,
                        max(0, probability[0] - 0.8) if probability[0] > 0.8 else 0
                    ]
                }
                
                fig = px.bar(
                    prob_data, 
                    x='Risk Level', 
                    y='Probability',
                    color='Risk Level',
                    color_discrete_map={
                        'Low': '#28a745',
                        'Medium': '#ffc107', 
                        'High': '#fd7e14',
                        'Critical': '#dc3545'
                    }
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Recommendations")
                recommendations = st.session_state.risk_assessor.get_recommendations(risk_level)
                for rec in recommendations:
                    st.write(f"• {rec}")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def show_historical_analysis():
    st.header("📈 Historical Data Analysis")
    
    # State and date range selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_state = st.selectbox("Select State:", list(INDIAN_STATES.keys()), key="hist_state")
    
    with col2:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    
    with col3:
        end_date = st.date_input("End Date", datetime.now())
    
    try:
        # Load historical data
        historical_data = st.session_state.data_loader.get_historical_data(
            selected_state, start_date, end_date
        )
        
        if historical_data is not None and not historical_data.empty:
            # Seasonal patterns
            st.subheader("Seasonal Rainfall Patterns")
            seasonal_fig = st.session_state.visualizer.create_seasonal_analysis(historical_data)
            st.plotly_chart(seasonal_fig, use_container_width=True)
            
            # Flood events timeline
            st.subheader("Historical Flood Events")
            flood_events = historical_data[historical_data['flood_occurred'] == 1]
            
            if not flood_events.empty:
                events_fig = st.session_state.visualizer.create_flood_events_timeline(flood_events)
                st.plotly_chart(events_fig, use_container_width=True)
                
                # Statistics
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    st.metric("Total Flood Events", len(flood_events))
                
                with col5:
                    avg_rainfall = flood_events['rainfall'].mean()
                    st.metric("Avg Rainfall during Floods", f"{avg_rainfall:.1f} mm")
                
                with col6:
                    max_rainfall = flood_events['rainfall'].max()
                    st.metric("Max Recorded Rainfall", f"{max_rainfall:.1f} mm")
            else:
                st.info("No historical flood events found for the selected period.")
        else:
            st.warning(f"No historical data available for {selected_state} in the selected date range.")
            st.info("Historical data will be displayed when available from official meteorological sources.")
    
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")

def show_model_performance():
    st.header("🎯 Model Performance Metrics")
    
    try:
        # Get model metrics
        metrics = st.session_state.model.get_model_metrics()
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
            
            # Confusion Matrix
            st.subheader("Model Confusion Matrix")
            if 'confusion_matrix' in metrics:
                conf_matrix_fig = st.session_state.visualizer.create_confusion_matrix(
                    metrics['confusion_matrix']
                )
                st.plotly_chart(conf_matrix_fig, use_container_width=True)
            
            # Feature Importance
            st.subheader("Feature Importance")
            if 'feature_importance' in metrics:
                importance_fig = st.session_state.visualizer.create_feature_importance_chart(
                    metrics['feature_importance'], MODEL_FEATURES
                )
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # ROC Curve
            st.subheader("ROC Curve")
            if 'roc_data' in metrics:
                roc_fig = st.session_state.visualizer.create_roc_curve(metrics['roc_data'])
                st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("Model metrics will be displayed after training completion.")
    
    except Exception as e:
        st.error(f"Error displaying model performance: {str(e)}")

def show_ai_insights():
    st.header("🤖 AI-Powered Climate Insights")
    
    # Check if Gemini is available
    gemini_available = os.getenv("GEMINI_API_KEY") is not None
    
    if not gemini_available:
        st.warning("⚠️ Gemini AI is not configured. Using synthetic data generation.")
        st.info("To enable AI insights, add your GEMINI_API_KEY in the Secrets panel.")
        return
    
    st.success("✅ AI Insights powered by Gemini")
    
    # State selection
    selected_state = st.selectbox("Select State for AI Analysis:", list(INDIAN_STATES.keys()), key="ai_state")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Risk Analysis")
        
        if st.button("Get AI Risk Assessment", type="primary"):
            with st.spinner("Analyzing current conditions with AI..."):
                try:
                    # Get current data
                    current_data = st.session_state.data_loader.get_current_weather_data(selected_state)
                    
                    if current_data:
                        # Get AI analysis
                        from data.gemini_climate_service import GeminiClimateService
                        gemini = GeminiClimateService()
                        analysis = gemini.get_flood_prediction_factors(selected_state, current_data)
                        
                        if analysis:
                            st.markdown("### AI Assessment")
                            st.write(analysis)
                            
                            # Display current conditions
                            st.markdown("### Current Conditions")
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("Rainfall", f"{current_data.get('rainfall', 0):.1f} mm")
                            with metrics_col2:
                                st.metric("River Level", f"{current_data.get('river_level', 0):.1f} m")
                            with metrics_col3:
                                st.metric("Temperature", f"{current_data.get('temperature', 0):.1f} °C")
                        else:
                            st.error("Could not get AI analysis. Please try again.")
                    else:
                        st.error("Could not fetch current weather data.")
                        
                except Exception as e:
                    st.error(f"Error getting AI insights: {str(e)}")
    
    with col2:
        st.subheader("📅 Weather Forecast")
        
        forecast_days = st.slider("Forecast Duration (days)", 3, 14, 7, key="forecast_days")
        
        if st.button("Generate AI Forecast", type="primary"):
            with st.spinner("Generating AI-powered forecast..."):
                try:
                    from data.gemini_climate_service import GeminiClimateService
                    gemini = GeminiClimateService()
                    forecast = gemini.generate_forecast(selected_state, forecast_days)
                    
                    if forecast:
                        st.markdown("### Forecast")
                        st.write(forecast)
                    else:
                        st.error("Could not generate forecast. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
    
    # Historical Insights
    st.markdown("---")
    st.subheader("📚 Historical Climate Insights")
    
    if st.button("Get Historical Analysis", type="secondary"):
        with st.spinner("Analyzing historical patterns with AI..."):
            try:
                from data.gemini_climate_service import GeminiClimateService
                gemini = GeminiClimateService()
                insights = gemini.get_historical_insights(selected_state, months=12)
                
                if insights:
                    st.write(insights)
                else:
                    st.error("Could not get historical insights. Please try again.")
                    
            except Exception as e:
                st.error(f"Error getting historical insights: {str(e)}")
    
    # Regional Comparison
    st.markdown("---")
    st.subheader("🗺️ Regional Risk Comparison")
    
    comparison_states = st.multiselect(
        "Select states to compare:",
        list(INDIAN_STATES.keys()),
        default=[selected_state],
        key="comparison_states"
    )
    
    if st.button("Compare Regions", type="secondary") and len(comparison_states) > 1:
        with st.spinner("Comparing regional flood risks with AI..."):
            try:
                from data.gemini_climate_service import GeminiClimateService
                gemini = GeminiClimateService()
                comparison = gemini.get_regional_comparison(comparison_states)
                
                if comparison:
                    st.write(comparison)
                else:
                    st.error("Could not get regional comparison. Please try again.")
                    
            except Exception as e:
                st.error(f"Error getting regional comparison: {str(e)}")

def show_risk_mapping():
    st.header("🗺️ Interactive Flood Risk Map")
    
    try:
        # Create base map of India
        india_map = folium.Map(
            location=[20.5937, 78.9629],  # Center of India
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Load regional risk data
        regional_data = st.session_state.data_loader.get_regional_risk_data()
        
        if regional_data is not None:
            # Add risk markers to map
            for _, row in regional_data.iterrows():
                risk_level = row.get('risk_level', 'Low')
                color = RISK_COLORS.get(risk_level, '#808080')
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=10,
                    popup=f"Region: {row['region']}<br>Risk: {risk_level}<br>Rainfall: {row.get('rainfall', 0):.1f}mm",
                    color=color,
                    fill=True,
                    fillColor=color,
                    weight=2
                ).add_to(india_map)
        
        # Display map
        map_data = st_folium(india_map, width=700, height=500)
        
        # Risk legend
        st.subheader("Risk Level Legend")
        legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
        
        with legend_col1:
            st.markdown("🟢 **Low Risk**")
        with legend_col2:
            st.markdown("🟡 **Medium Risk**")
        with legend_col3:
            st.markdown("🟠 **High Risk**")
        with legend_col4:
            st.markdown("🔴 **Critical Risk**")
        
        # Regional statistics
        if regional_data is not None:
            st.subheader("Regional Risk Summary")
            risk_summary = regional_data['risk_level'].value_counts()
            
            summary_fig = px.pie(
                values=risk_summary.values,
                names=risk_summary.index,
                title="Distribution of Risk Levels Across Regions",
                color=risk_summary.index,
                color_discrete_map={
                    'Low': '#28a745',
                    'Medium': '#ffc107',
                    'High': '#fd7e14',
                    'Critical': '#dc3545'
                }
            )
            st.plotly_chart(summary_fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying risk map: {str(e)}")
        st.info("Interactive flood risk map will be displayed when regional data is available.")

if __name__ == "__main__":
    main()
