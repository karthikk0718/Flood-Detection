import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

class SyntheticDataGenerator:
    """Generate realistic synthetic climate data for Indian states"""
    
    def __init__(self):
        # Monsoon patterns for different regions
        self.monsoon_patterns = {
            'West Coast': {'peak_months': [6, 7, 8], 'intensity': 1.5},
            'North': {'peak_months': [7, 8], 'intensity': 1.2},
            'Northeast': {'peak_months': [6, 7, 8, 9], 'intensity': 1.8},
            'South': {'peak_months': [10, 11], 'intensity': 1.3},
            'Central': {'peak_months': [7, 8], 'intensity': 1.0},
        }
        
        # State to region mapping
        self.state_regions = {
            'Kerala': 'West Coast',
            'Karnataka': 'West Coast',
            'Goa': 'West Coast',
            'Maharashtra': 'West Coast',
            'Gujarat': 'West Coast',
            'Assam': 'Northeast',
            'Meghalaya': 'Northeast',
            'Manipur': 'Northeast',
            'Nagaland': 'Northeast',
            'Mizoram': 'Northeast',
            'Tripura': 'Northeast',
            'Arunachal Pradesh': 'Northeast',
            'Punjab': 'North',
            'Haryana': 'North',
            'Himachal Pradesh': 'North',
            'Uttarakhand': 'North',
            'Jammu and Kashmir': 'North',
            'Ladakh': 'North',
            'Tamil Nadu': 'South',
            'Andhra Pradesh': 'South',
            'Telangana': 'South',
            'Madhya Pradesh': 'Central',
            'Chhattisgarh': 'Central',
            'Uttar Pradesh': 'Central',
            'Bihar': 'Central',
            'Jharkhand': 'Central',
            'Odisha': 'Central',
            'West Bengal': 'Central',
            'Rajasthan': 'Central',
            'Delhi': 'North',
            'Sikkim': 'Northeast',
        }
        
    def generate_historical_data(self, state: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate historical weather and flood data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        region = self.state_regions.get(state, 'Central')
        pattern = self.monsoon_patterns[region]
        
        data = []
        for date in date_range:
            month = date.month
            
            # Base rainfall with seasonal variation
            if month in pattern['peak_months']:
                base_rainfall = np.random.exponential(scale=20 * pattern['intensity'])
            else:
                base_rainfall = np.random.exponential(scale=5)
            
            # Add some extreme events (10% probability during monsoon)
            if month in pattern['peak_months'] and random.random() < 0.1:
                base_rainfall *= random.uniform(2, 5)
            
            rainfall = np.clip(base_rainfall, 0, 250)
            
            # River level based on rainfall
            base_river_level = 5 + (rainfall * 0.08) + np.random.normal(0, 0.5)
            river_level = np.clip(base_river_level, 1, 18)
            
            # Temperature variation
            if month in [12, 1, 2]:
                temperature = np.random.normal(20, 3)
            elif month in [3, 4, 5]:
                temperature = np.random.normal(32, 4)
            elif month in pattern['peak_months']:
                temperature = np.random.normal(28, 3)
            else:
                temperature = np.random.normal(25, 3)
            
            temperature = np.clip(temperature, 15, 45)
            
            # Other parameters
            humidity = np.clip(50 + rainfall * 0.5 + np.random.normal(0, 10), 30, 100)
            wind_speed = np.clip(np.random.exponential(12), 0, 60)
            
            # Determine if flood occurred
            flood_score = (rainfall / 50) * 0.4 + (river_level / 10) * 0.35 + (humidity / 100) * 0.15
            flood_occurred = 1 if flood_score > 0.7 and random.random() < 0.6 else 0
            
            # Severity for flood events
            severity = min(int(flood_score * 5), 5) if flood_occurred else 0
            
            data.append({
                'date': date,
                'rainfall': rainfall,
                'river_level': river_level,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'flood_occurred': flood_occurred,
                'severity': severity
            })
        
        return pd.DataFrame(data)
    
    def generate_current_weather(self, state: str) -> Dict:
        """Generate current weather conditions"""
        region = self.state_regions.get(state, 'Central')
        pattern = self.monsoon_patterns[region]
        current_month = datetime.now().month
        
        # Generate realistic current conditions
        if current_month in pattern['peak_months']:
            rainfall = np.clip(np.random.exponential(15 * pattern['intensity']), 0, 150)
            humidity = np.clip(np.random.normal(80, 10), 50, 100)
        else:
            rainfall = np.clip(np.random.exponential(3), 0, 50)
            humidity = np.clip(np.random.normal(60, 15), 30, 90)
        
        river_level = np.clip(5 + rainfall * 0.08 + np.random.normal(0, 1), 2, 15)
        
        # Temperature based on season
        if current_month in [12, 1, 2]:
            temperature = np.clip(np.random.normal(20, 4), 10, 30)
        elif current_month in [3, 4, 5]:
            temperature = np.clip(np.random.normal(32, 5), 25, 45)
        else:
            temperature = np.clip(np.random.normal(28, 4), 20, 40)
        
        wind_speed = np.clip(np.random.exponential(12), 5, 50)
        
        return {
            'rainfall': float(rainfall),
            'river_level': float(river_level),
            'temperature': float(temperature),
            'humidity': float(humidity),
            'wind_speed': float(wind_speed),
            'timestamp': datetime.now()
        }
    
    def generate_rainfall_trend(self, state: str, days: int = 7) -> pd.DataFrame:
        """Generate rainfall trend for last N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        region = self.state_regions.get(state, 'Central')
        pattern = self.monsoon_patterns[region]
        current_month = datetime.now().month
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        rainfall_data = []
        
        for date in dates:
            if current_month in pattern['peak_months']:
                rainfall = np.clip(np.random.exponential(12 * pattern['intensity']), 0, 120)
            else:
                rainfall = np.clip(np.random.exponential(5), 0, 40)
            
            rainfall_data.append({
                'date': date,
                'rainfall': rainfall
            })
        
        return pd.DataFrame(rainfall_data)
    
    def generate_river_data(self, state: str, hours: int = 48) -> pd.DataFrame:
        """Generate river level monitoring data"""
        end_time = datetime.now()
        timestamps = [end_time - timedelta(hours=i) for i in range(hours, 0, -1)]
        
        # Base level with gradual changes
        base_level = np.random.uniform(4, 7)
        levels = []
        flow_rates = []
        
        for i, ts in enumerate(timestamps):
            # Add trend and noise
            trend = 0.02 * np.sin(i / 6)  # Gentle oscillation
            noise = np.random.normal(0, 0.3)
            level = np.clip(base_level + trend + noise, 2, 15)
            levels.append(level)
            
            # Flow rate correlates with level
            flow_rate = np.clip(50 + (level - 5) * 20 + np.random.normal(0, 10), 10, 300)
            flow_rates.append(flow_rate)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'level': levels,
            'flow_rate': flow_rates
        })
    
    def generate_regional_risk_data(self, states: List[str]) -> pd.DataFrame:
        """Generate risk data for multiple regions"""
        regional_data = []
        
        # Coordinates for states (from config)
        from config.settings import INDIAN_STATES
        
        for state in states:
            if state not in INDIAN_STATES:
                continue
            
            lat, lon = INDIAN_STATES[state]
            current_weather = self.generate_current_weather(state)
            
            # Calculate risk level
            rainfall = current_weather['rainfall']
            river_level = current_weather['river_level']
            
            risk_score = (rainfall / 50) * 0.5 + (river_level / 10) * 0.35
            
            if risk_score > 0.8:
                risk_level = 'Critical'
            elif risk_score > 0.6:
                risk_level = 'High'
            elif risk_score > 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            regional_data.append({
                'region': state,
                'latitude': lat,
                'longitude': lon,
                'rainfall': rainfall,
                'river_level': river_level,
                'risk_level': risk_level
            })
        
        return pd.DataFrame(regional_data)
    
    def generate_national_flood_data(self, states: List[str]) -> pd.DataFrame:
        """Generate national overview of flood risks"""
        national_data = []
        
        for state in states:
            current_weather = self.generate_current_weather(state)
            rainfall = current_weather['rainfall']
            river_level = current_weather['river_level']
            
            risk_score = (rainfall / 50) * 0.5 + (river_level / 10) * 0.35
            
            if risk_score > 0.8:
                risk_level = 'Critical'
            elif risk_score > 0.6:
                risk_level = 'High'
            elif risk_score > 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            national_data.append({
                'state': state,
                'risk_level': risk_level,
                'rainfall': rainfall,
                'river_level': river_level
            })
        
        return pd.DataFrame(national_data)
