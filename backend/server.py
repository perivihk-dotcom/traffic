from fastapi import FastAPI, APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timezone
import numpy as np
import csv
import io
from fpdf import FPDF

from models.accident_model import AccidentRiskPredictor
from data.sample_data import generate_sample_accident_data, get_statistics, get_key_factors_by_severity

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'traffic_accident_db')]

# Create the main app
app = FastAPI(
    title="Road Traffic Accident Risk Prediction API",
    description="CNN-BiLSTM-Attention model for predicting traffic accident severity and identifying key risk factors",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize predictor
predictor = AccidentRiskPredictor()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== Pydantic Models ==============

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    hour: int = Field(default=12, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(default=0, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(default=6, ge=1, le=12, description="Month (1-12)")
    year: int = Field(default=2020, ge=2005, le=2025, description="Year")
    latitude: float = Field(default=51.5, description="Latitude")
    longitude: float = Field(default=-0.1, description="Longitude")
    speed_limit: int = Field(default=30, description="Speed limit (mph)")
    road_type: int = Field(default=6, ge=1, le=9, description="Road type code")
    junction_control: int = Field(default=0, ge=0, le=4, description="Junction control type")
    junction_detail: int = Field(default=0, ge=0, le=9, description="Junction detail")
    light_conditions: int = Field(default=1, ge=1, le=7, description="Light conditions code")
    weather_conditions: int = Field(default=1, ge=1, le=9, description="Weather conditions code")
    road_surface_conditions: int = Field(default=1, ge=1, le=7, description="Road surface conditions")
    urban_rural: int = Field(default=1, ge=1, le=2, description="1=Urban, 2=Rural")
    number_of_vehicles: int = Field(default=2, ge=1, le=10, description="Number of vehicles involved")
    number_of_casualties: int = Field(default=1, ge=1, le=10, description="Number of casualties")
    police_force: int = Field(default=1, ge=1, le=50, description="Police force code")
    vehicle_type: int = Field(default=3, ge=1, le=10, description="Vehicle type code")
    vehicle_manoeuvre: int = Field(default=1, ge=1, le=18, description="Vehicle manoeuvre")
    engine_capacity: int = Field(default=1500, ge=0, le=10000, description="Engine capacity (cc)")
    age_of_vehicle: int = Field(default=5, ge=0, le=50, description="Age of vehicle (years)")
    driver_age: int = Field(default=35, ge=16, le=100, description="Driver age")
    driver_sex: int = Field(default=1, ge=1, le=2, description="1=Male, 2=Female")
    journey_purpose: int = Field(default=1, ge=1, le=6, description="Journey purpose")
    pedestrian_crossing_human: int = Field(default=0, ge=0, le=2)
    pedestrian_crossing_physical: int = Field(default=0, ge=0, le=5)
    special_conditions: int = Field(default=0, ge=0, le=7)
    carriageway_hazards: int = Field(default=0, ge=0, le=7)
    first_road_class: int = Field(default=3, ge=1, le=6)
    second_road_class: int = Field(default=0, ge=0, le=6)
    road_type_detail: int = Field(default=6, ge=1, le=9)
    did_police_attend: int = Field(default=1, ge=1, le=2)

class PredictionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    severity: str
    risk_level: int
    probabilities: Dict[str, float]
    confidence: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_features: Dict[str, Any]

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    value: float
    category: str

class AccidentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    severity: int
    severity_name: str
    hour: int
    day_of_week: int
    month: int
    year: int
    latitude: float
    longitude: float
    speed_limit: int
    road_type_name: str
    weather_conditions_name: str
    light_conditions_name: str

class StatisticsResponse(BaseModel):
    total_accidents: int
    severity_distribution: Dict[str, int]
    hourly_distribution: List[Dict]
    daily_distribution: List[Dict]
    monthly_distribution: List[Dict]
    weather_distribution: List[Dict]
    road_type_distribution: List[Dict]
    speed_limit_distribution: List[Dict]

class ModelInfo(BaseModel):
    name: str
    architecture: str
    features: List[str]
    num_features: int
    classes: List[str]
    description: str

class BatchPredictionInput(BaseModel):
    records: List[PredictionInput]

class BatchPredictionResult(BaseModel):
    total_records: int
    predictions: List[PredictionResult]
    summary: Dict[str, Any]

class RouteCompareInput(BaseModel):
    route1: PredictionInput
    route2: PredictionInput

class RouteCompareResult(BaseModel):
    route1: Dict[str, Any]
    route2: Dict[str, Any]
    safer_route: int
    risk_difference: float
    recommendation: str

class AlertConfig(BaseModel):
    threshold_fatal: float = Field(default=0.3, ge=0, le=1)
    threshold_serious: float = Field(default=0.5, ge=0, le=1)

class RiskAlert(BaseModel):
    id: str
    severity: str
    risk_level: int
    alert_type: str
    message: str
    timestamp: datetime
    input_features: Dict[str, Any]

# ============== API Endpoints ==============

@api_router.get("/")
async def root():
    return {
        "message": "Road Traffic Accident Risk Prediction API",
        "version": "1.0.0",
        "model": "CNN-BiLSTM-Attention"
    }

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@api_router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the prediction model"""
    return ModelInfo(
        name="CNN-BiLSTM-Attention Traffic Accident Risk Predictor",
        architecture="Convolutional Neural Network + Bidirectional LSTM + Spatial-Temporal Attention",
        features=predictor.FEATURE_NAMES,
        num_features=len(predictor.FEATURE_NAMES),
        classes=predictor.SEVERITY_LABELS,
        description="Deep learning model for predicting traffic accident severity based on temporal, spatial, environmental, vehicle, and personnel factors."
    )

@api_router.post("/predict", response_model=PredictionResult)
async def predict_accident_risk(input_data: PredictionInput):
    """Predict accident severity risk based on input features"""
    try:
        # Convert input to dict
        features = input_data.model_dump()
        
        # Make prediction
        result = predictor.predict(features)
        
        # Create response
        prediction_result = PredictionResult(
            severity=result['severity'],
            risk_level=result['risk_level'],
            probabilities=result['probabilities'],
            confidence=result['confidence'],
            input_features=features
        )
        
        # Store prediction in database
        doc = prediction_result.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.predictions.insert_one(doc)
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict/explain", response_model=List[FeatureImportance])
async def explain_prediction(input_data: PredictionInput):
    """Get feature importance explanation for a prediction"""
    try:
        features = input_data.model_dump()
        importance = predictor.get_feature_importance(features)
        return importance[:15]  # Return top 15 factors
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@api_router.get("/predictions/history", response_model=List[PredictionResult])
async def get_prediction_history(limit: int = Query(default=50, le=100)):
    """Get recent prediction history"""
    predictions = await db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    for pred in predictions:
        if isinstance(pred['timestamp'], str):
            pred['timestamp'] = datetime.fromisoformat(pred['timestamp'])
    return predictions

@api_router.get("/data/sample", response_model=List[Dict])
async def get_sample_data(limit: int = Query(default=100, le=1000)):
    """Get sample accident data"""
    # Check if we have data in DB
    count = await db.accidents.count_documents({})
    
    if count == 0:
        # Generate and store sample data
        logger.info("Generating sample accident data...")
        sample_data = generate_sample_accident_data(1000)
        await db.accidents.insert_many(sample_data)
        logger.info(f"Inserted {len(sample_data)} sample records")
    
    accidents = await db.accidents.find({}, {"_id": 0}).to_list(limit)
    return accidents

@api_router.get("/data/statistics", response_model=StatisticsResponse)
async def get_data_statistics():
    """Get statistics about accident data"""
    # Ensure we have data
    count = await db.accidents.count_documents({})
    if count == 0:
        sample_data = generate_sample_accident_data(1000)
        await db.accidents.insert_many(sample_data)
    
    accidents = await db.accidents.find({}, {"_id": 0}).to_list(10000)
    stats = get_statistics(accidents)
    return stats

@api_router.get("/data/key-factors")
async def get_key_factors():
    """Get key factors affecting accident severity by level"""
    return get_key_factors_by_severity()

@api_router.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics (simulated based on paper results)"""
    return {
        "model_name": "CNN-BiLSTM-Attention",
        "dataset": "UK Road Safety (2015-2020)",
        "metrics": {
            "mae": 0.2475,
            "precision": 0.8191,
            "recall": 0.8782,
            "f1_score": 0.8476,
            "accuracy": 0.8534
        },
        "comparison": [
            {"model": "CNN-BiLSTM-Attention", "mae": 0.2475, "precision": 0.8191, "recall": 0.8782, "f1": 0.8476},
            {"model": "CNN+BiLSTM", "mae": 0.2612, "precision": 0.7985, "recall": 0.8523, "f1": 0.8245},
            {"model": "BiLSTM", "mae": 0.2834, "precision": 0.7756, "recall": 0.8245, "f1": 0.7992},
            {"model": "LSTM", "mae": 0.2956, "precision": 0.7623, "recall": 0.8156, "f1": 0.7881},
            {"model": "CNN", "mae": 0.3124, "precision": 0.7489, "recall": 0.7934, "f1": 0.7705},
            {"model": "MLP", "mae": 0.3267, "precision": 0.7312, "recall": 0.7756, "f1": 0.7527},
            {"model": "Random Forest", "mae": 0.3456, "precision": 0.7156, "recall": 0.7523, "f1": 0.7335},
            {"model": "SVM", "mae": 0.3678, "precision": 0.6923, "recall": 0.7234, "f1": 0.7075}
        ],
        "training_info": {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "cross_validation": "10-fold"
        }
    }

@api_router.get("/data/yearly")
async def get_yearly_data():
    """Get yearly trend of accidents"""
    count = await db.accidents.count_documents({})
    if count == 0:
        sample_data = generate_sample_accident_data(1000)
        await db.accidents.insert_many(sample_data)
    
    accidents = await db.accidents.find({}, {"_id": 0}).to_list(10000)
    
    # Yearly by severity
    yearly_severity = {}
    for acc in accidents:
        year = acc['year']
        if year not in yearly_severity:
            yearly_severity[year] = {'Slight': 0, 'Serious': 0, 'Fatal': 0, 'total': 0}
        yearly_severity[year][acc['severity_name']] += 1
        yearly_severity[year]['total'] += 1
    
    return {
        'yearly': [
            {'year': y, **yearly_severity[y]} for y in sorted(yearly_severity.keys())
        ]
    }

@api_router.get("/data/spatial")
async def get_spatial_data():
    """Get spatial distribution of accidents for map visualization"""
    count = await db.accidents.count_documents({})
    if count == 0:
        sample_data = generate_sample_accident_data(1000)
        await db.accidents.insert_many(sample_data)
    
    accidents = await db.accidents.find(
        {},
        {"_id": 0, "id": 1, "latitude": 1, "longitude": 1, "severity": 1, "severity_name": 1}
    ).to_list(1000)
    
    return {
        "total": len(accidents),
        "points": accidents
    }

@api_router.get("/data/temporal")
async def get_temporal_data():
    """Get temporal patterns of accidents"""
    count = await db.accidents.count_documents({})
    if count == 0:
        sample_data = generate_sample_accident_data(1000)
        await db.accidents.insert_many(sample_data)
    
    accidents = await db.accidents.find({}, {"_id": 0}).to_list(10000)
    
    # Hourly by severity
    hourly_severity = {h: {'Slight': 0, 'Serious': 0, 'Fatal': 0} for h in range(24)}
    for acc in accidents:
        hourly_severity[acc['hour']][acc['severity_name']] += 1
    
    # Day of week by severity
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_severity = {d: {'Slight': 0, 'Serious': 0, 'Fatal': 0} for d in range(7)}
    for acc in accidents:
        daily_severity[acc['day_of_week']][acc['severity_name']] += 1
    
    return {
        'hourly': [
            {'hour': h, **hourly_severity[h]} for h in range(24)
        ],
        'daily': [
            {'day': day_names[d], **daily_severity[d]} for d in range(7)
        ]
    }

@api_router.get("/data/environmental")
async def get_environmental_data():
    """Get environmental factors distribution"""
    count = await db.accidents.count_documents({})
    if count == 0:
        sample_data = generate_sample_accident_data(1000)
        await db.accidents.insert_many(sample_data)
    
    accidents = await db.accidents.find({}, {"_id": 0}).to_list(10000)
    
    # Weather by severity
    weather_severity = {}
    for acc in accidents:
        w = acc['weather_conditions_name']
        if w not in weather_severity:
            weather_severity[w] = {'Slight': 0, 'Serious': 0, 'Fatal': 0, 'total': 0}
        weather_severity[w][acc['severity_name']] += 1
        weather_severity[w]['total'] += 1
    
    # Light conditions by severity
    light_severity = {}
    for acc in accidents:
        l = acc['light_conditions_name']
        if l not in light_severity:
            light_severity[l] = {'Slight': 0, 'Serious': 0, 'Fatal': 0, 'total': 0}
        light_severity[l][acc['severity_name']] += 1
        light_severity[l]['total'] += 1
    
    # Road surface by severity
    surface_severity = {}
    for acc in accidents:
        s = acc['road_surface_name']
        if s not in surface_severity:
            surface_severity[s] = {'Slight': 0, 'Serious': 0, 'Fatal': 0, 'total': 0}
        surface_severity[s][acc['severity_name']] += 1
        surface_severity[s]['total'] += 1
    
    return {
        'weather': [{'condition': k, **v} for k, v in sorted(weather_severity.items(), key=lambda x: -x[1]['total'])],
        'light': [{'condition': k, **v} for k, v in sorted(light_severity.items(), key=lambda x: -x[1]['total'])],
        'surface': [{'condition': k, **v} for k, v in sorted(surface_severity.items(), key=lambda x: -x[1]['total'])]
    }

@api_router.delete("/data/reset")
async def reset_data():
    """Reset and regenerate sample data"""
    await db.accidents.delete_many({})
    sample_data = generate_sample_accident_data(1000)
    await db.accidents.insert_many(sample_data)
    return {"message": "Data reset successfully", "records": len(sample_data)}

# ============== New Endpoints ==============

@api_router.post("/predict/batch", response_model=BatchPredictionResult)
async def batch_predict(input_data: BatchPredictionInput):
    """Predict accident risk for multiple records"""
    try:
        predictions = []
        severity_counts = {'Slight': 0, 'Serious': 0, 'Fatal': 0}
        
        for record in input_data.records:
            features = record.model_dump()
            result = predictor.predict(features)
            
            prediction = PredictionResult(
                severity=result['severity'],
                risk_level=result['risk_level'],
                probabilities=result['probabilities'],
                confidence=result['confidence'],
                input_features=features
            )
            predictions.append(prediction)
            severity_counts[result['severity']] += 1
            
            # Store in database
            doc = prediction.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            doc['batch'] = True
            await db.predictions.insert_one(doc)
        
        total = len(predictions)
        summary = {
            'total': total,
            'severity_distribution': severity_counts,
            'slight_percentage': round(severity_counts['Slight'] / total * 100, 1) if total > 0 else 0,
            'serious_percentage': round(severity_counts['Serious'] / total * 100, 1) if total > 0 else 0,
            'fatal_percentage': round(severity_counts['Fatal'] / total * 100, 1) if total > 0 else 0,
            'average_confidence': round(sum(p.confidence for p in predictions) / total, 3) if total > 0 else 0
        }
        
        return BatchPredictionResult(
            total_records=total,
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@api_router.post("/predict/upload")
async def upload_and_predict(file: UploadFile = File(...)):
    """Upload CSV file and get predictions for all records"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        contents = await file.read()
        decoded = contents.decode('utf-8')
        reader = csv.DictReader(io.StringIO(decoded))
        
        predictions = []
        severity_counts = {'Slight': 0, 'Serious': 0, 'Fatal': 0}
        errors = []
        
        for idx, row in enumerate(reader):
            try:
                # Convert CSV row to prediction input
                features = {
                    'hour': int(row.get('hour', 12)),
                    'day_of_week': int(row.get('day_of_week', 0)),
                    'month': int(row.get('month', 6)),
                    'year': int(row.get('year', 2020)),
                    'latitude': float(row.get('latitude', 51.5)),
                    'longitude': float(row.get('longitude', -0.1)),
                    'speed_limit': int(row.get('speed_limit', 30)),
                    'road_type': int(row.get('road_type', 6)),
                    'junction_control': int(row.get('junction_control', 2)),
                    'light_conditions': int(row.get('light_conditions', 1)),
                    'weather_conditions': int(row.get('weather_conditions', 1)),
                    'road_surface_conditions': int(row.get('road_surface_conditions', 1)),
                    'urban_rural': int(row.get('urban_rural', 1)),
                    'number_of_vehicles': int(row.get('number_of_vehicles', 2)),
                    'number_of_casualties': int(row.get('number_of_casualties', 1)),
                    'vehicle_type': int(row.get('vehicle_type', 3)),
                    'engine_capacity': int(row.get('engine_capacity', 1500)),
                    'age_of_vehicle': int(row.get('age_of_vehicle', 5)),
                    'driver_age': int(row.get('driver_age', 35)),
                    'driver_sex': int(row.get('driver_sex', 1))
                }
                
                result = predictor.predict(features)
                prediction = {
                    'row': idx + 1,
                    'severity': result['severity'],
                    'risk_level': result['risk_level'],
                    'probabilities': result['probabilities'],
                    'confidence': result['confidence'],
                    'input_features': features
                }
                predictions.append(prediction)
                severity_counts[result['severity']] += 1
                
            except Exception as e:
                errors.append({'row': idx + 1, 'error': str(e)})
        
        total = len(predictions)
        return {
            'filename': file.filename,
            'total_records': total,
            'predictions': predictions,
            'errors': errors,
            'summary': {
                'severity_distribution': severity_counts,
                'slight_percentage': round(severity_counts['Slight'] / total * 100, 1) if total > 0 else 0,
                'serious_percentage': round(severity_counts['Serious'] / total * 100, 1) if total > 0 else 0,
                'fatal_percentage': round(severity_counts['Fatal'] / total * 100, 1) if total > 0 else 0,
                'average_confidence': round(sum(p['confidence'] for p in predictions) / total, 3) if total > 0 else 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload prediction failed: {str(e)}")

@api_router.post("/routes/compare", response_model=RouteCompareResult)
async def compare_routes(input_data: RouteCompareInput):
    """Compare two routes/locations for safety"""
    try:
        route1_features = input_data.route1.model_dump()
        route2_features = input_data.route2.model_dump()
        
        result = predictor.compare_routes(route1_features, route2_features)
        
        return RouteCompareResult(
            route1=result['route1'],
            route2=result['route2'],
            safer_route=result['safer_route'],
            risk_difference=result['risk_difference'],
            recommendation=result['recommendation']
        )
    except Exception as e:
        logger.error(f"Route comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Route comparison failed: {str(e)}")

@api_router.post("/alerts/check")
async def check_risk_alerts(input_data: PredictionInput, config: Optional[AlertConfig] = None):
    """Check if prediction triggers risk alerts"""
    try:
        if config is None:
            config = AlertConfig()
        
        features = input_data.model_dump()
        result = predictor.predict(features)
        
        alerts = []
        
        # Check for fatal risk
        if result['probabilities']['fatal'] >= config.threshold_fatal:
            alert = RiskAlert(
                id=str(uuid.uuid4()),
                severity='Fatal',
                risk_level=3,
                alert_type='HIGH_FATAL_RISK',
                message=f"High fatal accident risk detected ({result['probabilities']['fatal']*100:.1f}%). Exercise extreme caution!",
                timestamp=datetime.now(timezone.utc),
                input_features=features
            )
            alerts.append(alert)
            await db.alerts.insert_one(alert.model_dump())
        
        # Check for serious risk
        if result['probabilities']['serious'] >= config.threshold_serious:
            alert = RiskAlert(
                id=str(uuid.uuid4()),
                severity='Serious',
                risk_level=2,
                alert_type='HIGH_SERIOUS_RISK',
                message=f"Elevated serious accident risk detected ({result['probabilities']['serious']*100:.1f}%). Proceed with caution.",
                timestamp=datetime.now(timezone.utc),
                input_features=features
            )
            alerts.append(alert)
            await db.alerts.insert_one(alert.model_dump())
        
        return {
            'prediction': result,
            'alerts': [a.model_dump() for a in alerts],
            'has_alerts': len(alerts) > 0,
            'alert_count': len(alerts)
        }
    except Exception as e:
        logger.error(f"Alert check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Alert check failed: {str(e)}")

@api_router.get("/alerts/history")
async def get_alerts_history(limit: int = Query(default=50, le=200)):
    """Get recent risk alerts"""
    alerts = await db.alerts.find({}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return alerts

@api_router.get("/report/generate")
async def generate_report(prediction_id: Optional[str] = None):
    """Generate PDF report for predictions"""
    try:
        # Fetch prediction data
        if prediction_id:
            prediction = await db.predictions.find_one({"id": prediction_id}, {"_id": 0})
            predictions = [prediction] if prediction else []
        else:
            predictions = await db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).to_list(10)
        
        # Fetch statistics
        accidents = await db.accidents.find({}, {"_id": 0}).to_list(1000)
        stats = get_statistics(accidents) if accidents else {}
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 15, 'Traffic Accident Risk Prediction Report', 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        pdf.ln(10)
        
        # Model Information
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Model Information', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, 'Model: CNN-BiLSTM-Attention\nArchitecture: Convolutional Neural Network + Bidirectional LSTM + Spatial-Temporal Attention\nExplainability: DeepSHAP-based feature importance analysis')
        pdf.ln(5)
        
        # Performance Metrics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Model Performance', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, 'MAE: 0.2475 | Precision: 81.91% | Recall: 87.82% | F1 Score: 84.76% | Accuracy: 85.34%')
        pdf.ln(5)
        
        # Predictions Summary
        if predictions:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, f'Prediction Results ({len(predictions)} records)', 0, 1)
            pdf.set_font('Arial', '', 9)
            
            for i, pred in enumerate(predictions[:20], 1):  # Limit to 20 for PDF
                severity = pred.get('severity', 'N/A')
                confidence = pred.get('confidence', 0) * 100
                probs = pred.get('probabilities', {})
                
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, f'Prediction #{i}', 0, 1)
                pdf.set_font('Arial', '', 9)
                pdf.multi_cell(0, 5, f'Severity: {severity} | Confidence: {confidence:.1f}%\nProbabilities - Slight: {probs.get("slight", 0)*100:.1f}% | Serious: {probs.get("serious", 0)*100:.1f}% | Fatal: {probs.get("fatal", 0)*100:.1f}%')
                pdf.ln(2)
        
        # Dataset Statistics
        if stats:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Dataset Statistics', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            sev_dist = stats.get('severity_distribution', {})
            pdf.multi_cell(0, 6, f'Total Accidents: {stats.get("total_accidents", 0)}\nSlight: {sev_dist.get("Slight", 0)} | Serious: {sev_dist.get("Serious", 0)} | Fatal: {sev_dist.get("Fatal", 0)}')
            pdf.ln(5)
        
        # Key Factors
        key_factors = get_key_factors_by_severity()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Top Risk Factors (Global)', 0, 1)
        pdf.set_font('Arial', '', 9)
        
        for factor in key_factors.get('global', {}).get('factors', [])[:10]:
            pdf.cell(0, 5, f"- {factor['factor']}: {factor['importance']*100:.1f}%", 0, 1)
        
        # Recommendations
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Safety Recommendations', 0, 1)
        pdf.set_font('Arial', '', 10)
        recommendations = [
            "1. Enforce speed limits, especially in high-risk areas",
            "2. Increase awareness during peak hours (7-9 AM, 4-6 PM)",
            "3. Regular vehicle inspections for older vehicles",
            "4. Enhanced monitoring at high-risk junctions",
            "5. Avoid driving in adverse weather conditions"
        ]
        for rec in recommendations:
            pdf.cell(0, 6, rec, 0, 1)
        
        # Output PDF
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        return StreamingResponse(
            io.BytesIO(pdf_output),
            media_type='application/pdf',
            headers={'Content-Disposition': 'attachment; filename=accident_risk_report.pdf'}
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@api_router.get("/data/export")
async def export_predictions_csv():
    """Export predictions as CSV"""
    try:
        predictions = await db.predictions.find({}, {"_id": 0}).to_list(1000)
        
        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions to export")
        
        output = io.StringIO()
        fieldnames = ['id', 'timestamp', 'severity', 'risk_level', 'confidence', 
                      'prob_slight', 'prob_serious', 'prob_fatal']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for pred in predictions:
            writer.writerow({
                'id': pred.get('id', ''),
                'timestamp': pred.get('timestamp', ''),
                'severity': pred.get('severity', ''),
                'risk_level': pred.get('risk_level', ''),
                'confidence': pred.get('confidence', ''),
                'prob_slight': pred.get('probabilities', {}).get('slight', ''),
                'prob_serious': pred.get('probabilities', {}).get('serious', ''),
                'prob_fatal': pred.get('probabilities', {}).get('fatal', '')
            })
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type='text/csv',
            headers={'Content-Disposition': 'attachment; filename=predictions_export.csv'}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
