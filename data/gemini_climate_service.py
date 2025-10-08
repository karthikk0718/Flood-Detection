import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

# IMPORTANT: KEEP THIS COMMENT
# Using python_gemini blueprint for climate data generation

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ClimateData(BaseModel):
    rainfall: float
    river_level: float
    temperature: float
    humidity: float
    wind_speed: float
    flood_risk: str

class HistoricalPattern(BaseModel):
    season: str
    avg_rainfall: float
    flood_events: int
    risk_level: str

class GeminiClimateService:
    """Use Gemini AI to generate realistic climate data and predictions"""
    
    def __init__(self):
        self.model = "gemini-2.5-flash"
    
    def get_current_climate_data(self, state: str, region: str = "India") -> Optional[Dict]:
        """Get AI-generated realistic current climate data for a state"""
        try:
            current_month = datetime.now().strftime("%B")
            
            prompt = f"""Generate realistic current climate data for {state}, {region} in {current_month}.
            
Consider:
- Monsoon patterns (June-September is monsoon season in most of India)
- Regional climate variations (coastal, mountain, plains)
- Seasonal temperature ranges
- Typical rainfall patterns for this region and month

Provide realistic values for:
- rainfall (mm, current 24h)
- river_level (meters, typical range 2-15m)
- temperature (Celsius)
- humidity (percentage)
- wind_speed (km/h)
- flood_risk (Low/Medium/High/Critical based on conditions)

Respond with JSON only, no additional text."""

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClimateData,
                )
            )
            
            if response.text:
                data = json.loads(response.text)
                data['timestamp'] = datetime.now()
                return data
            
            return None
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def get_historical_insights(self, state: str, months: int = 12) -> Optional[str]:
        """Get AI-generated historical climate insights"""
        try:
            prompt = f"""Analyze historical climate and flood patterns for {state}, India over the past {months} months.
            
Provide insights on:
1. Seasonal rainfall patterns
2. Historical flood events and their causes
3. Most vulnerable months
4. Risk trends (increasing/decreasing)

Keep response concise and factual."""

            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return response.text if response.text else None
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def get_flood_prediction_factors(self, state: str, current_conditions: Dict) -> Optional[str]:
        """Get AI analysis of flood risk factors"""
        try:
            prompt = f"""Analyze flood risk for {state}, India with current conditions:
- Rainfall: {current_conditions.get('rainfall', 0)}mm
- River Level: {current_conditions.get('river_level', 5)}m
- Temperature: {current_conditions.get('temperature', 25)}°C
- Humidity: {current_conditions.get('humidity', 70)}%

Provide:
1. Risk assessment (Low/Medium/High/Critical)
2. Key contributing factors
3. Immediate recommendations

Be concise and practical."""

            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return response.text if response.text else None
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def generate_forecast(self, state: str, days: int = 7) -> Optional[str]:
        """Generate AI-powered weather forecast"""
        try:
            current_month = datetime.now().strftime("%B")
            
            prompt = f"""Generate a {days}-day weather forecast for {state}, India starting from {current_month}.
            
Consider:
- Current season and typical patterns
- Monsoon timing if applicable
- Regional climate characteristics

Provide:
- Daily rainfall expectations
- Flood risk outlook
- Key weather alerts if any

Format as a clear, structured forecast."""

            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return response.text if response.text else None
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
    
    def get_regional_comparison(self, states: List[str]) -> Optional[str]:
        """Get AI comparison of flood risks across regions"""
        try:
            states_str = ", ".join(states)
            current_month = datetime.now().strftime("%B")
            
            prompt = f"""Compare current flood risk levels across these Indian states in {current_month}: {states_str}
            
Provide:
1. Relative risk ranking
2. Key risk factors for each region
3. States requiring immediate attention

Be concise and focus on actionable insights."""

            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return response.text if response.text else None
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
