# Hospital Readmission Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive web-based healthcare analytics platform that leverages machine learning to predict hospital readmissions within 30 days, assess patient stress levels, and provide personalized health insights.

## 🎯 Features

### Core Functionality
- **30-Day Readmission Prediction**: Advanced ML models to predict hospital readmission risk
- **Stress Level Assessment**: Patient stress analysis based on lifestyle factors
- **Disease Risk Prediction**: Multi-label disease prediction system
- **Length of Stay Estimation**: Predict expected hospital stay duration
- **Interactive Analytics Dashboard**: Real-time data visualization with Plotly

### User Management
- **Secure Authentication**: User registration, login, and session management
- **Role-Based Access**: Support for patients and healthcare providers
- **Password Security**: Encrypted password storage with PBKDF2-SHA256

### Healthcare Features
- **Patient Checkup Management**: Comprehensive health data collection
- **Medical History Tracking**: Claims and procedure history
- **Personalized Risk Assessment**: Individual health risk profiling
- **Clinical Decision Support**: Evidence-based recommendations

## 🏗️ Architecture

### Technology Stack
- **Backend**: Flask (Python 3.8+)
- **Database**: SQLite with comprehensive schema
- **Machine Learning**: 
  - XGBoost for gradient boosting
  - Random Forest for ensemble learning
  - Scikit-learn for preprocessing and evaluation
  - TensorFlow/Keras for deep learning models
- **Frontend**: HTML5, CSS3, Bootstrap 4, JavaScript
- **Visualization**: Plotly.js for interactive charts
- **Deployment**: Flask development server (production-ready with WSGI)

### Machine Learning Models
1. **Readmission Prediction Model** (`enhanced_readmission_model.py`)
   - Target Accuracy: 80%+
   - Ensemble of Random Forest, XGBoost, Gradient Boosting, and Logistic Regression
   - Advanced feature engineering with 50+ derived features

2. **Stress Assessment Model**
   - Features: Steps, calorie intake, smoking, alcohol, sleep patterns
   - Multi-class classification for stress levels

3. **Disease Prediction Model**
   - Multi-label classification for disease risk
   - Integrates vital signs, lab results, and lifestyle factors

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/saishri-05/Readmission-Prediction-System.git
   cd Readmission-Prediction-System
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database**
   ```bash
   python -c "from app import init_db; init_db()"
   ```

5. **Prepare ML Models**
   Ensure the following model files are present:
   - `stress_model.pkl`
   - `label_encoder.pkl`
   - `disease_model.pkl`
   - `mlb.pkl`
   - `length_of_stay_model.pkl`
   - `readmission_30d_model.pkl`

6. **Run the Application**
   ```bash
   python app.py
   ```

7. **Access the Application**
   Open your browser and navigate to `http://localhost:5000`

## 🗄️ Database Schema

### Users Table
- `user_id`: Primary key
- `name`: User full name
- `age`: User age
- `email`: Unique email address
- `mobile_number`: Contact information
- `password`: Hashed password (PBKDF2-SHA256)

### CheckupData Table
- `checkup_id`: Primary key
- `user_id`: Foreign key to Users
- Vital signs: Age, gender, RBC/WBC count, BP, heart rate
- Lab results: Blood sugar, cholesterol, glucose, hemoglobin
- Lifestyle: Steps, calories, smoking, alcohol, sleep
- Claims: Service type, diagnosis/procedure codes, costs

### ResultData Table
- `result_id`: Primary key
- `user_id`, `checkup_id`: Foreign keys
- Predictions: Stress level, diseases, risk scores
- Readmission: Probability and classification

## 🚀 Usage

### For Patients
1. **Register/Login**: Create account or sign in
2. **Health Checkup**: Enter comprehensive health data
3. **View Results**: Access personalized risk assessments
4. **Track Progress**: Monitor health metrics over time

### For Healthcare Providers
1. **Patient Management**: Access patient records and predictions
2. **Risk Assessment**: Review readmission probabilities
3. **Clinical Insights**: Use ML predictions for decision support
4. **Analytics Dashboard**: Visualize population health trends

### API Endpoints
- `/login` - User authentication
- `/register` - New user registration  
- `/checkup` - Health data entry
- `/analysis` - Prediction results
- `/home` - Dashboard access

## 🧪 Model Performance

### Readmission Prediction
- **Target Accuracy**: 80%+
- **Features**: 50+ engineered features including:
  - Demographics and vital signs
  - Lifestyle risk factors
  - Comorbidity counts
  - Service complexity scores
  - Cost-based risk categories

### Feature Engineering
- Age-based risk categorization
- BMI and blood pressure classifications
- Diabetes and cholesterol indicators
- Lifestyle risk scoring
- Activity level assessment
- Service type risk mapping

## 🛡️ Security Features

- **Password Hashing**: PBKDF2-SHA256 with salt
- **Session Management**: Secure Flask sessions
- **Input Validation**: Form data sanitization
- **SQL Injection Protection**: Parameterized queries
- **Authentication Required**: Protected routes

## 📊 Dependencies

### Core Frameworks
- `Flask==3.1.2` - Web framework
- `scikit-learn==1.2.2` - Machine learning library
- `xgboost==2.1.4` - Gradient boosting framework
- `tensorflow==2.12.0` - Deep learning framework

### Data Processing
- `pandas==2.3.1` - Data manipulation
- `numpy==1.23.5` - Numerical computing
- `plotly==6.3.0` - Interactive visualizations

### Database & Security
- `sqlite3` (built-in) - Database engine
- `Werkzeug==3.1.3` - Password hashing utilities

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**SAI SHRI**
- Email: pnsaishri5@gmail.com
- GitHub: [@saishri-05](https://github.com/saishri-05)

## 🙏 Acknowledgments

- Healthcare professionals who provided domain expertise
- Open-source machine learning community
- Flask and scikit-learn development teams

## 🔮 Future Enhancements

- [ ] Integration with Electronic Health Records (EHR)
- [ ] Real-time monitoring dashboard
- [ ] Mobile application development
- [ ] Advanced deep learning models
- [ ] Multi-hospital deployment support
- [ ] Telemedicine integration
- [ ] Predictive analytics for population health

---

**Note**: This system is designed for research and educational purposes. Always consult qualified healthcare professionals for medical decisions.
