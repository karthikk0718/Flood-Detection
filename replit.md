# AI Flood Prediction System for India

## Overview

This is an AI-powered flood prediction system specifically designed for Indian states. The application uses machine learning to predict flood risks based on climate data and provides real-time monitoring, risk assessments, and visualizations through an interactive web interface built with Streamlit.

The system combines synthetic data generation with optional Gemini AI integration for realistic climate data, making it suitable for both demonstration and production use cases.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**: Streamlit web framework
- **Decision**: Use Streamlit for rapid UI development with Python-native data science components
- **Rationale**: Streamlit provides built-in support for data visualization, real-time updates, and session state management, eliminating the need for separate frontend/backend architecture
- **Components**:
  - Multi-page navigation (Dashboard, Real-time Prediction, AI Insights, Historical Analysis, Model Performance, Risk Mapping)
  - Session state management for model persistence and data caching
  - Interactive visualizations using Plotly and Folium for maps

### Backend Architecture

**Machine Learning Pipeline**:
- **Model Type**: RandomForest Classifier (with LogisticRegression as alternative)
- **Decision**: Ensemble learning approach for flood prediction
- **Features**: 8 key climate indicators (rainfall, river_level, temperature, humidity, wind_speed, previous_rainfall, soil_moisture, elevation)
- **Rationale**: Random Forest handles non-linear relationships well and provides feature importance metrics crucial for understanding flood risk factors

**Data Processing Architecture**:
- **Synthetic Data Generator**: Fallback system for demonstration and testing
  - Implements region-specific monsoon patterns for realistic Indian climate simulation
  - Supports 5 climate regions (West Coast, North, Northeast, South, Central)
  - Generates time-series data with seasonal variations
- **Gemini AI Integration**: Optional enhanced climate data generation
  - Uses structured output (Pydantic models) for type-safe data validation
  - Provides more realistic predictions based on actual climate patterns
  - Graceful fallback to synthetic data if API unavailable

**Risk Assessment System**:
- **Decision**: Four-tier risk classification (Low, Medium, High, Critical)
- **Approach**: Composite scoring based on normalized climate metrics
- **Output**: Risk level with actionable recommendations specific to each tier

### Data Storage

**Current Implementation**: In-memory data structures
- **Decision**: No persistent database in current architecture
- **Rationale**: Suitable for demonstration and real-time prediction use case
- **Data Sources**:
  - Synthetic data generation for training and testing
  - Optional API integration (OpenWeatherMap, Gemini AI)
  - Serialized ML models (joblib pickle files)

**Model Persistence**:
- Trained models saved as `flood_model.pkl`
- Feature scalers saved as `scaler.pkl`
- Allows model reuse across sessions without retraining

### Configuration Management

**Centralized Settings** (`config/settings.py`):
- **INDIAN_STATES**: Comprehensive mapping of 31 states/territories with geocoordinates
- **RISK_COLORS**: Standardized color scheme for risk visualization
- **MODEL_FEATURES**: Centralized feature list for consistency across data pipeline

**Decision**: Configuration-as-code approach
- **Rationale**: Makes regional customization easy and keeps domain knowledge centralized
- **Benefit**: Single source of truth for state mappings and risk thresholds

## External Dependencies

### Third-Party APIs

**Google Gemini AI API**:
- **Purpose**: Enhanced climate data generation with realistic patterns
- **Integration**: `google.genai` library with structured output (Pydantic models)
- **Authentication**: Environment variable `GEMINI_API_KEY`
- **Fallback**: System gracefully degrades to synthetic data if unavailable

**OpenWeatherMap API**:
- **Purpose**: Real-time weather data (configured but not fully implemented)
- **Authentication**: Environment variable `OPENWEATHER_API_KEY`
- **Status**: Placeholder for future real-data integration

### Python Libraries

**Core Data Science Stack**:
- `pandas`, `numpy`: Data manipulation and numerical computing
- `scikit-learn`: Machine learning models and preprocessing
- `joblib`: Model serialization

**Visualization**:
- `plotly`: Interactive charts and graphs
- `folium`: Geographic mapping for state-level risk visualization
- `streamlit-folium`: Folium integration with Streamlit

**Web Framework**:
- `streamlit`: Complete web application framework with built-in state management

**AI/ML**:
- `google-genai`: Gemini AI integration
- `pydantic`: Type validation for AI-generated structured outputs

### Environment Variables

Required for full functionality:
- `GEMINI_API_KEY`: Optional, enables AI-enhanced climate data
- `OPENWEATHER_API_KEY`: Optional, for real weather data integration

System designed to work with or without these credentials using synthetic data fallback.