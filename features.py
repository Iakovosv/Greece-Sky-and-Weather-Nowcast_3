"""
Enhanced feature engineering for pre-rain detection.
"""
import math
from typing import Any, Dict, List

def compute_derived_features(
    current: Dict[str, Any],
    buffer_recent: List[Dict[str, Any]],
    buffer_30m: List[Dict[str, Any]],
    now_utc: Any,
) -> Dict[str, float]:
    features = {}
    
    def safe_get(d, key, default=0.0):
        try:
            v = d.get(key)
            if v is None: return default
            f = float(v)
            return f if math.isfinite(f) else default
        except: return default
    
    # BASE FEATURES
    features['temperature'] = safe_get(current, 'temperature')
    features['humidity'] = safe_get(current, 'humidity')
    features['pressure'] = safe_get(current, 'pressure')
    features['windspeed'] = safe_get(current, 'windspeed')
    features['windgust'] = safe_get(current, 'windgust')
    features['solarradiation'] = safe_get(current, 'solarradiation')
    features['rainrate'] = safe_get(current, 'rainrate')
    features['rain_1m_mm'] = safe_get(current, 'rain_1m_mm')
    
    # SCALING CONSTANTS
    P_TYP = 3.0
    H_TYP = 10.0
    T_TYP = 2.0
    W_TYP = 5.0
    
    # 10-minute deltas (scaled)
    if len(buffer_recent) >= 2:
        prev = buffer_recent[-2]
        features['dP_10m_scaled'] = (safe_get(current, 'pressure') - safe_get(prev, 'pressure')) / P_TYP
        features['dRH_10m_scaled'] = (safe_get(current, 'humidity') - safe_get(prev, 'humidity')) / H_TYP
        features['dT_10m_scaled'] = (safe_get(current, 'temperature') - safe_get(prev, 'temperature')) / T_TYP
        features['dWind_10m_scaled'] = (safe_get(current, 'windspeed') - safe_get(prev, 'windspeed')) / W_TYP
    else:
        features['dP_10m_scaled'] = features['dRH_10m_scaled'] = features['dT_10m_scaled'] = features['dWind_10m_scaled'] = 0.0
    
    # 30-minute deltas (scaled)
    if len(buffer_recent) >= 11:
        prev = buffer_recent[-11]
        features['dP_30m_scaled'] = (safe_get(current, 'pressure') - safe_get(prev, 'pressure')) / (P_TYP * 2)
        features['dRH_30m_scaled'] = (safe_get(current, 'humidity') - safe_get(prev, 'humidity')) / (H_TYP * 2)
    else:
        features['dP_30m_scaled'] = features['dRH_30m_scaled'] = 0.0
    
    # Rolling statistics
    if buffer_30m:
        pressures = [safe_get(r, 'pressure') for r in buffer_30m]
        humidities = [safe_get(r, 'humidity') for r in buffer_30m]
        features['p_mean_30m_scaled'] = (sum(pressures) / len(pressures) - 1013) / P_TYP
        features['rh_mean_30m_scaled'] = (sum(humidities) / len(humidities) - 60) / H_TYP
        if len(pressures) >= 5:
            n = len(pressures)
            features['p_trend_30m'] = (pressures[-1] - pressures[0]) / n * 10 / P_TYP
            features['h_trend_30m'] = (humidities[-1] - humidities[0]) / n * 10 / H_TYP
        else:
            features['p_trend_30m'] = features['h_trend_30m'] = 0.0
        recent_5 = pressures[-5:] if len(pressures) >= 5 else pressures
        features['p_volatility_5m'] = (max(recent_5) - min(recent_5)) / 2 / P_TYP if len(recent_5) > 1 else 0.0
    else:
        features['p_mean_30m_scaled'] = features['rh_mean_30m_scaled'] = 0.0
        features['p_trend_30m'] = features['h_trend_30m'] = features['p_volatility_5m'] = 0.0
    
    # Composite indicators
    features['pre_rain_index'] = -features.get('dP_30m_scaled', 0.0) * 0.5 + features.get('dRH_30m_scaled', 0.0) * 0.5
    features['instability_index'] = features.get('rh_mean_30m_scaled', 0.0) * 0.3 - features.get('dP_10m_scaled', 0.0) * 0.4 + features.get('dWind_10m_scaled', 0.0) * 0.3
    
    # Cyclical
    hour = now_utc.hour + now_utc.minute / 60.0
    features['hour_sin'] = math.sin(2.0 * math.pi * (hour / 24.0))
    features['hour_cos'] = math.cos(2.0 * math.pi * (hour / 24.0))
    doy = now_utc.timetuple().tm_yday
    features['doy_sin'] = math.sin(2.0 * math.pi * (doy / 365.25))
    features['doy_cos'] = math.cos(2.0 * math.pi * (doy / 365.25))
    
    # Day/Night
    lux = safe_get(current, 'lux_klux', None)
    solar = safe_get(current, 'solar_wm2', None)
    features['is_day'] = 1.0 if (lux is not None and lux > 0.1) or (solar is not None and solar >= 10.0) else 0.0
    
    return features