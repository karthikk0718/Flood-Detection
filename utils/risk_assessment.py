from typing import List, Dict
import numpy as np

class RiskAssessment:
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
        
        self.recommendations = {
            'Low': [
                "Continue normal activities with weather monitoring",
                "Keep emergency supplies updated",
                "Stay informed about weather forecasts",
                "Ensure drainage systems are clear"
            ],
            'Medium': [
                "Monitor weather conditions closely",
                "Prepare emergency evacuation plan",
                "Stock up on essential supplies",
                "Avoid low-lying areas if possible",
                "Check on vulnerable community members"
            ],
            'High': [
                "Consider evacuation from flood-prone areas",
                "Move valuables to higher ground",
                "Avoid unnecessary travel",
                "Keep emergency communication devices ready",
                "Follow local authority advisories closely"
            ],
            'Critical': [
                "IMMEDIATE EVACUATION recommended",
                "Move to higher ground immediately",
                "Do not attempt to cross flooded roads",
                "Contact emergency services if needed",
                "Follow all official evacuation orders"
            ]
        }
    
    def assess_risk(self, rainfall: float, river_level: float, temperature: float) -> str:
        """
        Assess flood risk based on current conditions
        """
        try:
            # Normalize inputs to 0-1 scale
            rainfall_score = min(rainfall / 100.0, 1.0)  # Normalize by 100mm
            river_level_score = min(river_level / 15.0, 1.0)  # Normalize by 15m
            temp_score = max(0, (temperature - 20) / 25.0)  # Higher temp = more evaporation
            
            # Calculate composite risk score
            risk_score = (
                0.5 * rainfall_score +
                0.35 * river_level_score +
                0.15 * (1 - temp_score)  # Higher temp reduces risk slightly
            )
            
            return self.get_risk_level_from_score(risk_score)
            
        except Exception as e:
            return "Low"  # Default to low risk on error
    
    def get_risk_level_from_score(self, score: float) -> str:
        """Convert risk score to risk level"""
        if score >= self.risk_thresholds['high']:
            return "Critical"
        elif score >= self.risk_thresholds['medium']:
            return "High"
        elif score >= self.risk_thresholds['low']:
            return "Medium"
        else:
            return "Low"
    
    def get_risk_level_from_probability(self, probability: float) -> str:
        """Convert model probability to risk level"""
        return self.get_risk_level_from_score(probability)
    
    def get_recommendations(self, risk_level: str) -> List[str]:
        """Get safety recommendations for risk level"""
        return self.recommendations.get(risk_level, self.recommendations['Low'])
    
    def calculate_regional_risk(self, regional_data: Dict) -> Dict[str, str]:
        """
        Calculate risk levels for multiple regions
        """
        try:
            regional_risks = {}
            
            for region, data in regional_data.items():
                rainfall = data.get('rainfall', 0)
                river_level = data.get('river_level', 5)
                temperature = data.get('temperature', 25)
                
                risk_level = self.assess_risk(rainfall, river_level, temperature)
                regional_risks[region] = risk_level
            
            return regional_risks
            
        except Exception as e:
            return {}
    
    def get_alert_message(self, risk_level: str, location: str = "") -> str:
        """
        Generate alert message for given risk level
        """
        messages = {
            'Low': f"🟢 LOW RISK: Normal conditions in {location}. Continue regular monitoring.",
            'Medium': f"🟡 MEDIUM RISK: Elevated flood risk in {location}. Stay alert and prepared.",
            'High': f"🟠 HIGH RISK: Significant flood risk in {location}. Take precautionary measures immediately.",
            'Critical': f"🔴 CRITICAL RISK: Extreme flood danger in {location}. Evacuate immediately if advised."
        }
        
        return messages.get(risk_level, messages['Low'])
    
    def calculate_risk_trend(self, historical_scores: List[float]) -> str:
        """
        Calculate if risk is increasing, decreasing, or stable
        """
        try:
            if len(historical_scores) < 2:
                return "Stable"
            
            recent_avg = np.mean(historical_scores[-3:])  # Last 3 readings
            older_avg = np.mean(historical_scores[:-3])   # Earlier readings
            
            if recent_avg > older_avg + 0.1:
                return "Increasing"
            elif recent_avg < older_avg - 0.1:
                return "Decreasing"
            else:
                return "Stable"
                
        except Exception:
            return "Stable"
    
    def get_evacuation_zones(self, risk_data: Dict[str, float]) -> List[str]:
        """
        Identify areas that should consider evacuation
        """
        evacuation_zones = []
        
        for region, risk_score in risk_data.items():
            risk_level = self.get_risk_level_from_score(risk_score)
            if risk_level in ['High', 'Critical']:
                evacuation_zones.append(region)
        
        return evacuation_zones
