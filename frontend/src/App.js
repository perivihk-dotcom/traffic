import { useState, useEffect, useCallback } from "react";
import "@/App.css";
import axios from "axios";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from "recharts";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Color palette
const COLORS = {
  slight: "#10b981",
  serious: "#f59e0b",
  fatal: "#ef4444",
  primary: "#3b82f6",
  secondary: "#8b5cf6",
  accent: "#06b6d4"
};

const SEVERITY_COLORS = [COLORS.slight, COLORS.serious, COLORS.fatal];

// Navigation Component
const Navigation = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: "📊" },
    { id: "predict", label: "Risk Prediction", icon: "🎯" },
    { id: "analysis", label: "Data Analysis", icon: "📈" },
    { id: "factors", label: "Key Factors", icon: "🔑" }
  ];

  return (
    <nav className="bg-slate-900 border-b border-slate-700" data-testid="navigation">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">🚗</span>
            <h1 className="text-xl font-bold text-white">Traffic Accident Risk Predictor</h1>
          </div>
          <div className="flex space-x-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                data-testid={`nav-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-blue-600 text-white"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

// Stats Card Component
const StatCard = ({ title, value, subtitle, icon, color = "blue" }) => {
  const colorClasses = {
    blue: "from-blue-500 to-blue-600",
    green: "from-green-500 to-green-600",
    yellow: "from-yellow-500 to-yellow-600",
    red: "from-red-500 to-red-600",
    purple: "from-purple-500 to-purple-600"
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-slate-400 text-sm font-medium">{title}</p>
          <p className="text-3xl font-bold text-white mt-1">{value}</p>
          {subtitle && <p className="text-slate-500 text-sm mt-1">{subtitle}</p>}
        </div>
        <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${colorClasses[color]} flex items-center justify-center text-2xl`}>
          {icon}
        </div>
      </div>
    </div>
  );
};

// Dashboard Component
const Dashboard = ({ stats, temporal }) => {
  if (!stats) return <LoadingSpinner />;

  const pieData = [
    { name: "Slight", value: stats.severity_distribution?.Slight || 0 },
    { name: "Serious", value: stats.severity_distribution?.Serious || 0 },
    { name: "Fatal", value: stats.severity_distribution?.Fatal || 0 }
  ];

  return (
    <div className="space-y-6" data-testid="dashboard">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Dashboard Overview</h2>
        <span className="text-slate-400">Based on UK Road Safety Data (2015-2020)</span>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Accidents"
          value={stats.total_accidents?.toLocaleString() || 0}
          subtitle="In dataset"
          icon="🚨"
          color="blue"
        />
        <StatCard
          title="Slight Injuries"
          value={stats.severity_distribution?.Slight?.toLocaleString() || 0}
          subtitle={`${((stats.severity_distribution?.Slight / stats.total_accidents) * 100).toFixed(1)}%`}
          icon="🩹"
          color="green"
        />
        <StatCard
          title="Serious Injuries"
          value={stats.severity_distribution?.Serious?.toLocaleString() || 0}
          subtitle={`${((stats.severity_distribution?.Serious / stats.total_accidents) * 100).toFixed(1)}%`}
          icon="🏥"
          color="yellow"
        />
        <StatCard
          title="Fatal Accidents"
          value={stats.severity_distribution?.Fatal?.toLocaleString() || 0}
          subtitle={`${((stats.severity_distribution?.Fatal / stats.total_accidents) * 100).toFixed(1)}%`}
          icon="⚠️"
          color="red"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Severity Distribution Pie Chart */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Severity Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={SEVERITY_COLORS[index]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Distribution */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Hourly Accident Pattern</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={stats.hourly_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" fontSize={12} />
              <YAxis stroke="#9ca3af" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                labelStyle={{ color: "#fff" }}
              />
              <Bar dataKey="count" fill={COLORS.primary} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Weekly & Monthly Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Daily Pattern</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={stats.daily_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="day" stroke="#9ca3af" fontSize={11} />
              <YAxis stroke="#9ca3af" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
              />
              <Bar dataKey="count" fill={COLORS.secondary} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Monthly Pattern</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={stats.monthly_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="month" stroke="#9ca3af" fontSize={12} />
              <YAxis stroke="#9ca3af" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
              />
              <Line type="monotone" dataKey="count" stroke={COLORS.accent} strokeWidth={3} dot={{ fill: COLORS.accent }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Temporal by Severity */}
      {temporal && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Hourly Accidents by Severity</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={temporal.hourly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" fontSize={12} />
              <YAxis stroke="#9ca3af" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
              />
              <Legend />
              <Bar dataKey="Slight" stackId="a" fill={COLORS.slight} />
              <Bar dataKey="Serious" stackId="a" fill={COLORS.serious} />
              <Bar dataKey="Fatal" stackId="a" fill={COLORS.fatal} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// Prediction Form Component
const PredictionForm = ({ onPredict, loading }) => {
  const [formData, setFormData] = useState({
    hour: 12,
    day_of_week: 0,
    month: 6,
    year: 2020,
    latitude: 51.5,
    longitude: -0.1,
    speed_limit: 30,
    road_type: 6,
    junction_control: 2,
    light_conditions: 1,
    weather_conditions: 1,
    road_surface_conditions: 1,
    urban_rural: 1,
    number_of_vehicles: 2,
    number_of_casualties: 1,
    vehicle_type: 3,
    engine_capacity: 1500,
    age_of_vehicle: 5,
    driver_age: 35,
    driver_sex: 1
  });

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(formData);
  };

  const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

  return (
    <form onSubmit={handleSubmit} className="space-y-6" data-testid="prediction-form">
      {/* Time Factors */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <span className="mr-2">🕐</span> Time Factors
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Hour (0-23)</label>
            <input
              type="number"
              min="0"
              max="23"
              value={formData.hour}
              onChange={(e) => handleChange("hour", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              data-testid="input-hour"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Day of Week</label>
            <select
              value={formData.day_of_week}
              onChange={(e) => handleChange("day_of_week", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-day"
            >
              {days.map((day, i) => (
                <option key={i} value={i}>{day}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Month</label>
            <select
              value={formData.month}
              onChange={(e) => handleChange("month", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-month"
            >
              {months.map((month, i) => (
                <option key={i} value={i + 1}>{month}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Year</label>
            <input
              type="number"
              min="2015"
              max="2025"
              value={formData.year}
              onChange={(e) => handleChange("year", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-year"
            />
          </div>
        </div>
      </div>

      {/* Location & Road */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <span className="mr-2">📍</span> Location & Road Factors
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Latitude</label>
            <input
              type="number"
              step="0.01"
              value={formData.latitude}
              onChange={(e) => handleChange("latitude", parseFloat(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-latitude"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Longitude</label>
            <input
              type="number"
              step="0.01"
              value={formData.longitude}
              onChange={(e) => handleChange("longitude", parseFloat(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-longitude"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Speed Limit (mph)</label>
            <select
              value={formData.speed_limit}
              onChange={(e) => handleChange("speed_limit", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-speed"
            >
              {[20, 30, 40, 50, 60, 70].map(limit => (
                <option key={limit} value={limit}>{limit} mph</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Area Type</label>
            <select
              value={formData.urban_rural}
              onChange={(e) => handleChange("urban_rural", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-area"
            >
              <option value={1}>Urban</option>
              <option value={2}>Rural</option>
            </select>
          </div>
        </div>
      </div>

      {/* Environmental Factors */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <span className="mr-2">🌤️</span> Environmental Factors
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Light Conditions</label>
            <select
              value={formData.light_conditions}
              onChange={(e) => handleChange("light_conditions", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-light"
            >
              <option value={1}>Daylight</option>
              <option value={4}>Darkness - lights lit</option>
              <option value={5}>Darkness - lights unlit</option>
              <option value={6}>Darkness - no lighting</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Weather</label>
            <select
              value={formData.weather_conditions}
              onChange={(e) => handleChange("weather_conditions", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-weather"
            >
              <option value={1}>Fine (no high winds)</option>
              <option value={2}>Raining (no high winds)</option>
              <option value={3}>Snowing (no high winds)</option>
              <option value={4}>Fine + high winds</option>
              <option value={5}>Raining + high winds</option>
              <option value={6}>Snowing + high winds</option>
              <option value={7}>Fog or mist</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Road Surface</label>
            <select
              value={formData.road_surface_conditions}
              onChange={(e) => handleChange("road_surface_conditions", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-surface"
            >
              <option value={1}>Dry</option>
              <option value={2}>Wet or damp</option>
              <option value={3}>Snow</option>
              <option value={4}>Frost or ice</option>
              <option value={5}>Flood</option>
            </select>
          </div>
        </div>
      </div>

      {/* Vehicle & Personnel */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <span className="mr-2">🚗</span> Vehicle & Personnel Factors
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Number of Vehicles</label>
            <input
              type="number"
              min="1"
              max="10"
              value={formData.number_of_vehicles}
              onChange={(e) => handleChange("number_of_vehicles", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-vehicles"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Engine Capacity (cc)</label>
            <input
              type="number"
              min="500"
              max="6000"
              step="100"
              value={formData.engine_capacity}
              onChange={(e) => handleChange("engine_capacity", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-engine"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Vehicle Age (years)</label>
            <input
              type="number"
              min="0"
              max="30"
              value={formData.age_of_vehicle}
              onChange={(e) => handleChange("age_of_vehicle", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-vehicle-age"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Driver Age</label>
            <input
              type="number"
              min="17"
              max="100"
              value={formData.driver_age}
              onChange={(e) => handleChange("driver_age", parseInt(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500"
              data-testid="input-driver-age"
            />
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        data-testid="predict-button"
      >
        {loading ? (
          <><span className="animate-spin mr-2">⏳</span> Predicting...</>
        ) : (
          <><span className="mr-2">🎯</span> Predict Accident Risk</>
        )}
      </button>
    </form>
  );
};

// Prediction Result Component
const PredictionResult = ({ result, importance }) => {
  if (!result) return null;

  const severityConfig = {
    Slight: { color: "bg-green-500", icon: "✅", desc: "Low severity - minor injuries expected" },
    Serious: { color: "bg-yellow-500", icon: "⚠️", desc: "Medium severity - significant injuries possible" },
    Fatal: { color: "bg-red-500", icon: "🚨", desc: "High severity - life-threatening risk" }
  };

  const config = severityConfig[result.severity];

  const probData = [
    { name: "Slight", value: result.probabilities.slight * 100 },
    { name: "Serious", value: result.probabilities.serious * 100 },
    { name: "Fatal", value: result.probabilities.fatal * 100 }
  ];

  return (
    <div className="space-y-6" data-testid="prediction-result">
      {/* Main Result Card */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Prediction Result</h3>
          <span className="text-slate-400 text-sm">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
        </div>
        
        <div className={`${config.color} rounded-xl p-6 text-center`}>
          <span className="text-4xl">{config.icon}</span>
          <h2 className="text-3xl font-bold text-white mt-2">{result.severity}</h2>
          <p className="text-white/80 mt-1">{config.desc}</p>
          <p className="text-white/60 text-sm mt-2">Risk Level: {result.risk_level}/3</p>
        </div>

        {/* Probability Bars */}
        <div className="mt-6 space-y-3">
          <h4 className="text-sm font-medium text-slate-400">Probability Distribution</h4>
          {probData.map((item, i) => (
            <div key={item.name} className="flex items-center space-x-3">
              <span className="text-sm text-slate-400 w-16">{item.name}</span>
              <div className="flex-1 bg-slate-700 rounded-full h-4 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${item.value}%`, backgroundColor: SEVERITY_COLORS[i] }}
                />
              </div>
              <span className="text-sm text-white w-16 text-right">{item.value.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Importance */}
      {importance && importance.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Top Contributing Factors</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={importance.slice(0, 10)} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#9ca3af" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="feature" stroke="#9ca3af" width={120} fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                formatter={(v) => `${(v * 100).toFixed(2)}%`}
              />
              <Bar dataKey="importance" fill={COLORS.primary} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// Risk Prediction Page
const RiskPrediction = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [importance, setImportance] = useState(null);

  const handlePredict = async (formData) => {
    setLoading(true);
    try {
      // Get prediction
      const predResponse = await axios.post(`${API}/predict`, formData);
      setResult(predResponse.data);

      // Get feature importance
      const impResponse = await axios.post(`${API}/predict/explain`, formData);
      setImportance(impResponse.data);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6" data-testid="risk-prediction">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Risk Prediction</h2>
          <p className="text-slate-400 mt-1">Enter accident parameters to predict severity risk using CNN-BiLSTM-Attention model</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <PredictionForm onPredict={handlePredict} loading={loading} />
        </div>
        <div>
          {result ? (
            <PredictionResult result={result} importance={importance} />
          ) : (
            <div className="bg-slate-800 rounded-xl p-12 border border-slate-700 text-center">
              <span className="text-6xl">🎯</span>
              <h3 className="text-xl font-semibold text-white mt-4">Ready to Predict</h3>
              <p className="text-slate-400 mt-2">Fill in the form and click predict to see the accident risk assessment</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Data Analysis Page
const DataAnalysis = ({ stats, environmental }) => {
  if (!stats || !environmental) return <LoadingSpinner />;

  return (
    <div className="space-y-6" data-testid="data-analysis">
      <h2 className="text-2xl font-bold text-white">Data Analysis</h2>

      {/* Weather Analysis */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4">Weather Conditions Impact</h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={environmental.weather}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="condition" stroke="#9ca3af" fontSize={10} angle={-45} textAnchor="end" height={80} />
            <YAxis stroke="#9ca3af" fontSize={12} />
            <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
            <Legend />
            <Bar dataKey="Slight" stackId="a" fill={COLORS.slight} />
            <Bar dataKey="Serious" stackId="a" fill={COLORS.serious} />
            <Bar dataKey="Fatal" stackId="a" fill={COLORS.fatal} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Light & Surface Conditions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Light Conditions</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={environmental.light}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="condition" stroke="#9ca3af" fontSize={9} angle={-30} textAnchor="end" height={70} />
              <YAxis stroke="#9ca3af" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
              <Legend />
              <Bar dataKey="Slight" fill={COLORS.slight} />
              <Bar dataKey="Serious" fill={COLORS.serious} />
              <Bar dataKey="Fatal" fill={COLORS.fatal} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Road Surface</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={environmental.surface}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="condition" stroke="#9ca3af" fontSize={10} angle={-30} textAnchor="end" height={70} />
              <YAxis stroke="#9ca3af" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
              <Legend />
              <Bar dataKey="Slight" fill={COLORS.slight} />
              <Bar dataKey="Serious" fill={COLORS.serious} />
              <Bar dataKey="Fatal" fill={COLORS.fatal} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Speed Limit Analysis */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4">Speed Limit Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={stats.speed_limit_distribution}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="limit" stroke="#9ca3af" tickFormatter={(v) => `${v} mph`} />
            <YAxis stroke="#9ca3af" fontSize={12} />
            <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
            <Bar dataKey="count" fill={COLORS.secondary} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// Key Factors Page
const KeyFactors = ({ keyFactors }) => {
  const [selectedSeverity, setSelectedSeverity] = useState("global");

  if (!keyFactors) return <LoadingSpinner />;

  const severityOptions = [
    { id: "global", label: "Global Factors", icon: "🌐" },
    { id: "slight", label: "Slight Accidents", icon: "🩹" },
    { id: "serious", label: "Serious Accidents", icon: "🏥" },
    { id: "fatal", label: "Fatal Accidents", icon: "⚠️" }
  ];

  const renderFactors = () => {
    if (selectedSeverity === "global") {
      const data = keyFactors.global.factors;
      return (
        <div className="space-y-4">
          <p className="text-slate-400 mb-4">{keyFactors.global.description}</p>
          <ResponsiveContainer width="100%" height={500}>
            <BarChart data={data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#9ca3af" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="factor" stroke="#9ca3af" width={130} fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                formatter={(v) => `${(v * 100).toFixed(2)}%`}
              />
              <Bar dataKey="importance" fill={COLORS.primary} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    } else {
      const data = keyFactors[selectedSeverity]?.top_factors || [];
      return (
        <div className="space-y-4">
          {data.map((factor, i) => (
            <div key={i} className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-white">#{i + 1} {factor.factor}</span>
                <span className="text-blue-400 font-mono">{(factor.importance * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-600 rounded-full h-2 mb-2">
                <div
                  className="bg-blue-500 h-2 rounded-full"
                  style={{ width: `${factor.importance * 100 * 5}%` }}
                />
              </div>
              <p className="text-slate-400 text-sm">{factor.description}</p>
            </div>
          ))}
        </div>
      );
    }
  };

  return (
    <div className="space-y-6" data-testid="key-factors">
      <div>
        <h2 className="text-2xl font-bold text-white">Key Risk Factors</h2>
        <p className="text-slate-400 mt-1">Factors identified using DeepSHAP explainability analysis</p>
      </div>

      {/* Severity Selector */}
      <div className="flex flex-wrap gap-2">
        {severityOptions.map((option) => (
          <button
            key={option.id}
            onClick={() => setSelectedSeverity(option.id)}
            data-testid={`factor-${option.id}`}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              selectedSeverity === option.id
                ? "bg-blue-600 text-white"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            <span className="mr-2">{option.icon}</span>
            {option.label}
          </button>
        ))}
      </div>

      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        {renderFactors()}
      </div>

      {/* Mitigation Recommendations */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <span className="mr-2">💡</span> Mitigation Recommendations
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-700/50 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">🚦 Speed Management</h4>
            <p className="text-slate-400 text-sm">Enforce speed limits especially in high-risk areas. Speed is the #1 factor in accident severity.</p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">🕐 Time-Based Alerts</h4>
            <p className="text-slate-400 text-sm">Increase awareness during peak accident hours (7-9 AM, 4-6 PM) and late night periods.</p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">🔧 Vehicle Inspection</h4>
            <p className="text-slate-400 text-sm">Regular vehicle inspections, especially for older vehicles lacking modern safety features.</p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">📍 Location Monitoring</h4>
            <p className="text-slate-400 text-sm">Enhanced monitoring at high-risk junctions and locations identified through spatial analysis.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Loading Spinner
const LoadingSpinner = () => (
  <div className="flex items-center justify-center h-64" data-testid="loading">
    <div className="text-center">
      <div className="animate-spin text-4xl mb-4">⏳</div>
      <p className="text-slate-400">Loading data...</p>
    </div>
  </div>
);

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [stats, setStats] = useState(null);
  const [temporal, setTemporal] = useState(null);
  const [environmental, setEnvironmental] = useState(null);
  const [keyFactors, setKeyFactors] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch all data in parallel
      const [statsRes, temporalRes, envRes, factorsRes] = await Promise.all([
        axios.get(`${API}/data/statistics`),
        axios.get(`${API}/data/temporal`),
        axios.get(`${API}/data/environmental`),
        axios.get(`${API}/data/key-factors`)
      ]);

      setStats(statsRes.data);
      setTemporal(temporalRes.data);
      setEnvironmental(envRes.data);
      setKeyFactors(factorsRes.data);
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const renderContent = () => {
    if (loading && activeTab !== "predict") {
      return <LoadingSpinner />;
    }

    switch (activeTab) {
      case "dashboard":
        return <Dashboard stats={stats} temporal={temporal} />;
      case "predict":
        return <RiskPrediction />;
      case "analysis":
        return <DataAnalysis stats={stats} environmental={environmental} />;
      case "factors":
        return <KeyFactors keyFactors={keyFactors} />;
      default:
        return <Dashboard stats={stats} temporal={temporal} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-900">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="max-w-7xl mx-auto px-4 py-6">
        {renderContent()}
      </main>
      <footer className="bg-slate-900 border-t border-slate-800 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
          <p>Road Traffic Accident Risk Prediction System | CNN-BiLSTM-Attention Model with DeepSHAP</p>
          <p className="mt-1">Based on UK Road Safety Dataset (2015-2020)</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
