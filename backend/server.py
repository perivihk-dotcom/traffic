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
