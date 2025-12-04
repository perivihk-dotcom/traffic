import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict

def generate_sample_accident_data(n_samples: int = 1000) -> List[Dict]:
    """
    Generate synthetic traffic accident data based on UK Road Safety dataset characteristics.
    The data follows patterns identified in the research paper.
    """
    np.random.seed(42)
    random.seed(42)
    
    accidents = []
    
    # UK geographic bounds
    lat_range = (50.0, 58.0)
    lon_range = (-6.0, 2.0)
    
    # Time patterns from paper (accidents more common during rush hours)
    hour_weights = [
        0.02, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5
        0.05, 0.08, 0.09, 0.06, 0.05, 0.05,  # 6-11
        0.06, 0.05, 0.05, 0.07, 0.08, 0.09,  # 12-17
        0.06, 0.04, 0.03, 0.03, 0.02, 0.02   # 18-23
    ]
    
    # Day of week patterns (Friday has more accidents)
    day_weights = [0.13, 0.14, 0.14, 0.14, 0.16, 0.15, 0.14]
    
    # Speed limits common in UK
    speed_limits = [20, 30, 40, 50, 60, 70]
    speed_weights = [0.05, 0.45, 0.15, 0.10, 0.20, 0.05]
    
    # Road types
    road_types = {
        1: 'Roundabout',
        2: 'One way street',
        3: 'Dual carriageway',
        6: 'Single carriageway',
        7: 'Slip road',
        9: 'Unknown'
    }
    
    # Weather conditions
    weather_conditions = {
        1: 'Fine no high winds',
        2: 'Raining no high winds',
        3: 'Snowing no high winds',
        4: 'Fine + high winds',
        5: 'Raining + high winds',
        6: 'Snowing + high winds',
        7: 'Fog or mist',
        8: 'Other',
        9: 'Unknown'
    }
    weather_weights = [0.65, 0.18, 0.02, 0.05, 0.04, 0.01, 0.02, 0.02, 0.01]
    
    # Light conditions
    light_conditions = {
        1: 'Daylight',
        4: 'Darkness - lights lit',
        5: 'Darkness - lights unlit',
        6: 'Darkness - no lighting',
        7: 'Darkness - lighting unknown'
    }
    light_weights = [0.70, 0.15, 0.05, 0.05, 0.05]
    
    # Surface conditions
    surface_conditions = {
        1: 'Dry',
        2: 'Wet or damp',
        3: 'Snow',
        4: 'Frost or ice',
        5: 'Flood over 3cm',
        6: 'Oil or diesel',
        7: 'Mud'
    }
    surface_weights = [0.60, 0.30, 0.02, 0.03, 0.01, 0.02, 0.02]
    
    # Vehicle types
    vehicle_types = {
        1: 'Pedal cycle',
        2: 'Motorcycle',
        3: 'Car',
        4: 'Taxi',
        5: 'Bus/Coach',
        8: 'Van',
        9: 'HGV',
        10: 'Other'
    }
    vehicle_weights = [0.05, 0.10, 0.60, 0.05, 0.03, 0.10, 0.05, 0.02]
    
    for i in range(n_samples):
        # Time features
        year = random.randint(2015, 2020)
        month = random.randint(1, 12)
        day_of_week = random.choices(range(7), weights=day_weights)[0]
        hour = random.choices(range(24), weights=hour_weights)[0]
        
        # Spatial features
        latitude = random.uniform(*lat_range)
        longitude = random.uniform(*lon_range)
        speed_limit = random.choices(speed_limits, weights=speed_weights)[0]
        road_type = random.choice(list(road_types.keys()))
        urban_rural = 1 if random.random() < 0.7 else 2  # 1=Urban, 2=Rural
        junction_control = random.randint(0, 4)
        junction_detail = random.randint(0, 9)
        
        # Environmental features
        weather = random.choices(list(weather_conditions.keys()), weights=weather_weights)[0]
        light = random.choices(list(light_conditions.keys()), weights=light_weights)[0]
        surface = random.choices(list(surface_conditions.keys()), weights=surface_weights)[0]
        
        # Vehicle features
        num_vehicles = random.choices([1, 2, 3, 4, 5], weights=[0.20, 0.55, 0.15, 0.07, 0.03])[0]
        vehicle_type = random.choices(list(vehicle_types.keys()), weights=vehicle_weights)[0]
        engine_capacity = random.randint(500, 4000)
        age_of_vehicle = random.randint(0, 25)
        vehicle_manoeuvre = random.randint(1, 18)
        
        # Personnel features
        num_casualties = random.choices([1, 2, 3, 4, 5], weights=[0.75, 0.15, 0.06, 0.03, 0.01])[0]
        driver_age = random.randint(17, 85)
        driver_sex = random.choice([1, 2])  # 1=Male, 2=Female
        journey_purpose = random.randint(1, 6)
        
        # Calculate severity based on risk factors (simplified model)
        risk_score = 0
        
        # Speed limit impact
        if speed_limit >= 60:
            risk_score += 2
        elif speed_limit >= 40:
            risk_score += 1
        
        # Weather impact
        if weather in [3, 5, 6, 7]:  # Bad weather
            risk_score += 1.5
        elif weather in [2, 4]:  # Moderate
            risk_score += 0.5
        
        # Light conditions impact
        if light in [5, 6]:  # Darkness without lights
            risk_score += 1.5
        elif light in [4, 7]:  # Darkness with lights
            risk_score += 0.5
        
        # Number of vehicles
        if num_vehicles >= 3:
            risk_score += 1
        
        # Surface conditions
        if surface in [3, 4, 5]:  # Snow, ice, flood
            risk_score += 1
        
        # Time of day
        if hour in [0, 1, 2, 3, 4, 23]:  # Late night
            risk_score += 0.5
        
        # Add some randomness
        risk_score += random.gauss(0, 1)
        
        # Determine severity
        if risk_score >= 4:
            severity = 3  # Fatal
        elif risk_score >= 2:
            severity = 2  # Serious
        else:
            severity = 1  # Slight
        
        accident = {
            'id': f'ACC{year}{i:06d}',
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'year': year,
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'speed_limit': speed_limit,
            'road_type': road_type,
            'road_type_name': road_types[road_type],
            'junction_control': junction_control,
            'junction_detail': junction_detail,
            'light_conditions': light,
            'light_conditions_name': light_conditions[light],
            'weather_conditions': weather,
            'weather_conditions_name': weather_conditions[weather],
            'road_surface_conditions': surface,
            'road_surface_name': surface_conditions[surface],
            'urban_rural': urban_rural,
            'urban_rural_name': 'Urban' if urban_rural == 1 else 'Rural',
            'number_of_vehicles': num_vehicles,
            'number_of_casualties': num_casualties,
            'police_force': random.randint(1, 50),
            'vehicle_type': vehicle_type,
            'vehicle_type_name': vehicle_types[vehicle_type],
            'vehicle_manoeuvre': vehicle_manoeuvre,
            'engine_capacity': engine_capacity,
            'age_of_vehicle': age_of_vehicle,
            'driver_age': driver_age,
            'driver_sex': driver_sex,
            'driver_sex_name': 'Male' if driver_sex == 1 else 'Female',
            'journey_purpose': journey_purpose,
            'pedestrian_crossing_human': random.randint(0, 2),
            'pedestrian_crossing_physical': random.randint(0, 5),
            'special_conditions': random.randint(0, 7),
            'carriageway_hazards': random.randint(0, 7),
            'first_road_class': random.randint(1, 6),
            'second_road_class': random.randint(0, 6),
            'road_type_detail': road_type,
            'did_police_attend': random.choice([1, 2]),
            'severity': severity,
            'severity_name': ['', 'Slight', 'Serious', 'Fatal'][severity]
        }
        
        accidents.append(accident)
    
    return accidents


def get_statistics(accidents: List[Dict]) -> Dict:
    """Calculate statistics from accident data"""
    total = len(accidents)
    
    # Severity distribution
    severity_counts = {'Slight': 0, 'Serious': 0, 'Fatal': 0}
    for acc in accidents:
        severity_counts[acc['severity_name']] += 1
    
    # Hourly distribution
    hourly = {i: 0 for i in range(24)}
    for acc in accidents:
        hourly[acc['hour']] += 1
    
    # Daily distribution
    daily = {i: 0 for i in range(7)}
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for acc in accidents:
        daily[acc['day_of_week']] += 1
    
    # Monthly distribution
    monthly = {i: 0 for i in range(1, 13)}
    for acc in accidents:
        monthly[acc['month']] += 1
    
    # Weather distribution
    weather_dist = {}
    for acc in accidents:
        w = acc['weather_conditions_name']
        weather_dist[w] = weather_dist.get(w, 0) + 1
    
    # Road type distribution
    road_dist = {}
    for acc in accidents:
        r = acc['road_type_name']
        road_dist[r] = road_dist.get(r, 0) + 1
    
    # Speed limit distribution
    speed_dist = {}
    for acc in accidents:
        s = acc['speed_limit']
        speed_dist[s] = speed_dist.get(s, 0) + 1
    
    return {
        'total_accidents': total,
        'severity_distribution': severity_counts,
        'hourly_distribution': [{'hour': h, 'count': hourly[h]} for h in range(24)],
        'daily_distribution': [{'day': day_names[d], 'count': daily[d]} for d in range(7)],
        'monthly_distribution': [{'month': m, 'count': monthly[m]} for m in range(1, 13)],
        'weather_distribution': [{'condition': k, 'count': v} for k, v in sorted(weather_dist.items(), key=lambda x: -x[1])],
        'road_type_distribution': [{'type': k, 'count': v} for k, v in sorted(road_dist.items(), key=lambda x: -x[1])],
        'speed_limit_distribution': [{'limit': k, 'count': v} for k, v in sorted(speed_dist.items())]
    }


def get_key_factors_by_severity() -> Dict:
    """Return key factors affecting different severity levels (from paper)"""
    return {
        'slight': {
            'top_factors': [
                {'factor': 'Hour of day', 'importance': 0.18, 'description': 'Peak hours have higher slight accident rates'},
                {'factor': 'Day of week', 'importance': 0.15, 'description': 'Weekdays show different patterns than weekends'},
                {'factor': 'Speed limit', 'importance': 0.14, 'description': '30mph zones have most slight accidents'},
                {'factor': 'Road type', 'importance': 0.12, 'description': 'Single carriageways are most common'},
                {'factor': 'Number of vehicles', 'importance': 0.11, 'description': 'Two-vehicle accidents are most frequent'}
            ]
        },
        'serious': {
            'top_factors': [
                {'factor': 'Speed limit', 'importance': 0.22, 'description': 'Higher speed limits correlate with serious injuries'},
                {'factor': 'Engine capacity', 'importance': 0.16, 'description': 'Larger vehicles can cause more serious harm'},
                {'factor': 'Junction control', 'importance': 0.14, 'description': 'Uncontrolled junctions are more dangerous'},
                {'factor': 'Location (Lat/Lon)', 'importance': 0.13, 'description': 'Certain areas have higher serious accident rates'},
                {'factor': 'Hour of day', 'importance': 0.10, 'description': 'Night hours see more serious accidents'}
            ]
        },
        'fatal': {
            'top_factors': [
                {'factor': 'Number of vehicles', 'importance': 0.20, 'description': 'Multi-vehicle collisions are more likely fatal'},
                {'factor': 'Speed limit', 'importance': 0.19, 'description': 'High-speed roads have more fatalities'},
                {'factor': 'Day of week', 'importance': 0.15, 'description': 'Weekend nights are particularly dangerous'},
                {'factor': 'Age of vehicle', 'importance': 0.12, 'description': 'Older vehicles lack modern safety features'},
                {'factor': 'Hour of day', 'importance': 0.11, 'description': 'Late night hours are most dangerous'}
            ]
        },
        'global': {
            'description': 'Top 15 factors affecting overall accident severity',
            'factors': [
                {'rank': 1, 'factor': 'Speed limit', 'importance': 0.095},
                {'rank': 2, 'factor': 'Number of vehicles', 'importance': 0.088},
                {'rank': 3, 'factor': 'Hour', 'importance': 0.082},
                {'rank': 4, 'factor': 'Month', 'importance': 0.075},
                {'rank': 5, 'factor': 'Year', 'importance': 0.068},
                {'rank': 6, 'factor': 'Engine capacity', 'importance': 0.065},
                {'rank': 7, 'factor': 'Road type', 'importance': 0.062},
                {'rank': 8, 'factor': 'Junction control', 'importance': 0.058},
                {'rank': 9, 'factor': 'Latitude', 'importance': 0.055},
                {'rank': 10, 'factor': 'Longitude', 'importance': 0.052},
                {'rank': 11, 'factor': 'Light conditions', 'importance': 0.048},
                {'rank': 12, 'factor': 'Weather conditions', 'importance': 0.045},
                {'rank': 13, 'factor': 'Day of week', 'importance': 0.042},
                {'rank': 14, 'factor': 'Age of vehicle', 'importance': 0.038},
                {'rank': 15, 'factor': 'Driver age', 'importance': 0.035}
            ]
        }
    }
