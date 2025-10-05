# 🚀 Exoplanet Discovery Platform

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118+-green.svg)](https://fastapi.tiangolo.com)
[![NASA Data](https://img.shields.io/badge/Data-NASA%20Missions-orange.svg)](https://nasa.gov)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A sophisticated web application and API platform for predicting and discovering exoplanets using machine learning models trained on NASA's space mission data. This project leverages cutting-edge ensemble algorithms to identify potential exoplanet candidates from stellar observations.

## 🌟 Features

### 🔍 **Exoplanet Prediction Engine**
- **Random Forest Classifier** trained on real NASA mission data
- **Binary classification**: CONFIRMED vs CANDIDATE exoplanets
- **High accuracy** prediction with comprehensive model metrics
- **Real-time API** for instant predictions

### 🌐 **Interactive Web Application**
- **Beautiful space-themed UI** with animated starfield backgrounds
- **Search & Discovery**: Browse and search exoplanet databases
- **Detailed Views**: Comprehensive exoplanet information and characteristics
- **Interactive Maps**: Visualize exoplanet locations and distributions
- **Community Features**: Share discoveries and insights

### 🔬 **Scientific Foundation**
- **NASA Mission Data**: Trained on authentic space telescope observations
- **Research-Based**: Implementation follows peer-reviewed methodologies
- **Ensemble Methods**: Utilizes Random Forest and Gradient Boosting algorithms
- **Feature Engineering**: Advanced preprocessing and imputation techniques

## 🏗️ Architecture

```
📁 nasa-space-apps-challenge/
├── 🖥️  frontend/           # Flask web application
│   ├── templates/          # HTML templates with space theme
│   └── static/            # CSS, JS, and assets
├── ⚡ backend/             # FastAPI ML prediction service
│   ├── app/               # API application core
│   └── routes/            # API endpoints
├── 🤖 ml/                 # Machine learning models
│   ├── models/            # Trained model files
│   └── raw-data/          # NASA dataset processing
├── 🐳 Docker configs       # Containerization setup
└── 📊 Data processing     # Demo cases and validation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Docker & Docker Compose (for production deployment)
- UV package manager (recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/nasa-space-apps-challenge.git
cd nasa-space-apps-challenge
```

### 2. Install Dependencies
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
# Start the Flask frontend
python main.py
```

### 4. Start the ML API (separate terminal)
```bash
# Navigate to backend directory
cd backend
python -m app.main
```

### 5. Access the Application
- **Web App**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/

## 🔧 Production Deployment

### Docker Deployment
```bash
# Build and deploy with Docker Compose
docker-compose up -d

# Access via configured domains:
# - Frontend: https://exoplanets.study
# - API: https://api.exoplanets.study
```

### Manual Deployment
```bash
# Build the application
./deploy.sh

# Or run individual services
python -m backend.app.main    # API service
python main.py                # Web application
```

## 🤖 Machine Learning Models

### Random Forest Classifier
- **Purpose**: Binary classification of exoplanet candidates
- **Training Data**: NASA space mission observations
- **Features**: Stellar characteristics, orbital parameters, photometric data
- **Performance**: Detailed metrics available via `/models` API endpoint

### Model Features
- **Preprocessing Pipeline**: Automatic missing value imputation
- **Feature Scaling**: StandardScaler for optimal performance
- **Cross-Validation**: RepeatedKFold for robust evaluation
- **Hyperparameter Tuning**: RandomizedSearchCV optimization

### Prediction Categories
- **CONFIRMED (1)**: High-confidence exoplanet detections
- **CANDIDATE (0)**: Potential exoplanets requiring further analysis

## 📡 API Reference

### Core Endpoints

#### Health Check
```http
GET /
```
Returns API status and welcome message.

#### Model Information
```http
GET /models
```
Returns trained model metrics and performance statistics.

#### Prediction Endpoint
```http
POST /predict
Content-Type: application/json

{
  "stellar_mass": 1.0,
  "stellar_radius": 1.0,
  "orbital_period": 365.25,
  "planet_radius": 1.0,
  "equilibrium_temperature": 288,
  ...
}
```

### Response Format
```json
{
  "prediction": 1,
  "probability": 0.87,
  "confidence": "high",
  "model_version": "1.0.0"
}
```

## 🔬 Scientific Background

This project implements methodologies from recent exoplanet detection research, specifically:

> **Reference**: Luz, T. S. F., Braga, R. A. S., & Ribeiro, E. R. (2024). *Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification*. Electronics, 13(19), 3950.

### Data Sources
- **NASA Exoplanet Archive**: Confirmed exoplanet catalog
- **Kepler Space Telescope**: Transit photometry data
- **TESS Mission**: Time-series stellar observations
- **Ground-based Surveys**: Radial velocity measurements

### Methodology
1. **Data Preprocessing**: Clean and normalize NASA mission datasets
2. **Feature Engineering**: Extract relevant stellar and orbital parameters
3. **Model Training**: Ensemble methods with cross-validation
4. **Validation**: Performance metrics and confusion matrix analysis
5. **Deployment**: Real-time prediction API

## 🎯 Use Cases

### 🔬 **Researchers & Scientists**
- Analyze large datasets of stellar observations
- Validate exoplanet candidates from space missions
- Compare model predictions with known discoveries

### 🎓 **Educators & Students**
- Learn about exoplanet detection methods
- Explore machine learning applications in astronomy
- Interactive visualization of exoplanet characteristics

### 🌍 **Public & Enthusiasts**
- Discover fascinating exoplanets and their properties
- Understand the scale and diversity of planetary systems
- Contribute to citizen science initiatives

## 🛠️ Development

### Project Structure
```python
# Core components
app.py              # Gradio demo interface
main.py             # Flask web application
backend/app/main.py # FastAPI ML service
ml/models/          # Machine learning implementations
```

### Adding New Models
1. Implement model class in `ml/models/`
2. Add training pipeline and evaluation metrics
3. Export trained model using joblib
4. Register model endpoint in FastAPI routes

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📊 Performance Metrics

The Random Forest model achieves the following performance on NASA validation data:

- **Accuracy**: Available via API `/models` endpoint
- **Precision**: Detailed metrics in model output
- **Recall**: Comprehensive evaluation results
- **F1-Score**: Balanced performance measurement

## 🤝 NASA Space Apps Challenge

This project was developed for the **NASA Space Apps Challenge**, focusing on:

- **Challenge Theme**: Exoplanet detection and characterization
- **Data Utilization**: NASA's open science datasets
- **Innovation**: Machine learning approaches to space exploration
- **Community Impact**: Educational and research applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA** for providing open access to space mission data
- **NASA Space Apps Challenge** for inspiring innovation in space exploration
- **Scientific Community** for research methodologies and validation
- **Open Source Contributors** for the tools and frameworks used

## 🔗 Links

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/)
- [Kepler Space Telescope](https://www.nasa.gov/kepler)
- [TESS Mission](https://www.nasa.gov/tess-transiting-exoplanet-survey-satellite/)

---

<div align="center">
<strong>🌌 Exploring the cosmos, one exoplanet at a time 🌌</strong>
<br><br>
<em>Made with ❤️ for the NASA Space Apps Challenge</em>
</div>
