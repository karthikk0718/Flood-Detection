import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
import requests
import json
from data.synthetic_data_generator import SyntheticDataGenerator
from data.gemini_climate_service import GeminiClimateService
from config.settings import INDIAN_STATES

class DataLoader:
    def __init__(self):
        self.api_base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")
        self.data_generator = SyntheticDataGenerator()
        self.gemini_service = GeminiClimateService()
        self.use_gemini = os.getenv("GEMINI_API_KEY") is not None
        
        # Indian states coordinates for mapping
        self.state_coordinates = INDIAN_STATES
    
    def get_current_weather_data(self, state: str) -> Optional[Dict[str, Any]]:
        """
        Get current weather data for a specific state
        Uses Gemini AI if available, otherwise synthetic data
        """
        try:
            if state not in self.state_coordinates:
                return None
            
            # Try Gemini first if available
            if self.use_gemini:
                gemini_data = self.gemini_service.get_current_climate_data(state)
                if gemini_data:
                    return gemini_data
            
            # Fallback to synthetic data generator
            return self.data_generator.generate_current_weather(state)
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_rainfall_trend(self, state: str, days: int = 7) -> Optional[pd.DataFrame]:
        """
        Get rainfall trend data for specified number of days
        """
        try:
            return self.data_generator.generate_rainfall_trend(state, days)
            
        except Exception as e:
            print(f"Error fetching rainfall trend: {e}")
            return None
    
    def get_river_level_data(self, state: str) -> Optional[pd.DataFrame]:
        """
        Get river level monitoring data
        """
        try:
            return self.data_generator.generate_river_data(state, hours=48)
            
        except Exception as e:
            print(f"Error fetching river level data: {e}")
            return None
    
    def get_national_flood_data(self) -> Optional[pd.DataFrame]:
        """
        Get national flood monitoring data
        """
        try:
            # Get data for major states
            major_states = list(self.state_coordinates.keys())[:15]
            return self.data_generator.generate_national_flood_data(major_states)
            
        except Exception as e:
            print(f"Error fetching national flood data: {e}")
            return None
    
    def get_historical_data(self, state: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get historical weather and flood data for analysis
        """
        try:
            return self.data_generator.generate_historical_data(state, start_date, end_date)
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def get_regional_risk_data(self) -> Optional[pd.DataFrame]:
        """
        Get regional flood risk data for mapping
        """
        try:
            # Get data for all states
            all_states = list(self.state_coordinates.keys())
            return self.data_generator.generate_regional_risk_data(all_states)
            
        except Exception as e:
            print(f"Error fetching regional risk data: {e}")
            return None
    
    def _estimate_river_level(self, weather_data: Dict) -> float:
        """
        Estimate river level based on weather conditions
        This is a simplified estimation for demonstration
        """
        try:
            rainfall = weather_data.get('rain', {}).get('1h', 0)
            humidity = weather_data['main']['humidity']
            
            # Simple estimation based on recent rainfall and humidity
            base_level = 5.0  # meters
            rainfall_factor = rainfall * 0.1  # 10cm per mm of rainfall
            humidity_factor = (humidity - 50) * 0.02  # Humidity effect
            
            estimated_level = base_level + rainfall_factor + humidity_factor
            return max(0, estimated_level)
            
        except Exception:
            return 5.0  # Default river level
    
    def validate_api_connection(self) -> bool:
        """
        Validate if API connection is working
        """
        try:
            if self.api_key == "your_api_key_here":
                return False
            
            # Test API call
            url = f"{self.api_base_url}/weather"
            params = {
                'q': 'Delhi,IN',
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
            
        except Exception:
            return False
