"""
Enhanced Hospital Readmission Prediction Model
Objective: Achieve 80%+ accuracy in predicting 30-day readmissions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedReadmissionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_names = []
        self.performance_metrics = {}
        
    def engineer_features(self, df):
        """Advanced feature engineering for readmission prediction"""
        df_enhanced = df.copy()
        
        # Age-based risk categories
        df_enhanced['age_risk_category'] = pd.cut(df_enhanced['age'], 
                                                 bins=[0, 40, 65, 80, 100], 
                                                 labels=['low', 'medium', 'high', 'very_high'])
        
        # BMI categories
        df_enhanced['bmi_category'] = pd.cut(df_enhanced['bmi'], 
                                           bins=[0, 18.5, 25, 30, 50], 
                                           labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Blood pressure categories
        df_enhanced['bp_category'] = 'normal'
        df_enhanced.loc[(df_enhanced['systole'] >= 140) | (df_enhanced['diastole'] >= 90), 'bp_category'] = 'hypertensive'
        df_enhanced.loc[(df_enhanced['systole'] < 90) | (df_enhanced['diastole'] < 60), 'bp_category'] = 'hypotensive'
        
        # Diabetes indicator
        df_enhanced['diabetes_indicator'] = (df_enhanced['blood_sugar'] > 126).astype(int)
        
        # High cholesterol indicator
        df_enhanced['high_cholesterol'] = (df_enhanced['cholesterol'] > 200).astype(int)
        
        # Comorbidity count (from diseases column)
        if 'diseases' in df_enhanced.columns:
            df_enhanced['comorbidity_count'] = df_enhanced['diseases'].str.count(',') + 1
            df_enhanced['comorbidity_count'] = df_enhanced['comorbidity_count'].fillna(0)
        else:
            df_enhanced['comorbidity_count'] = 0
        
        # Lifestyle risk score
        lifestyle_risk = 0
        if 'smoking' in df_enhanced.columns:
            lifestyle_risk += df_enhanced['smoking'] * 2
        if 'alcohol' in df_enhanced.columns:
            lifestyle_risk += df_enhanced['alcohol'] * 1
        if 'sleep_pattern' in df_enhanced.columns:
            lifestyle_risk += (df_enhanced['sleep_pattern'] < 6).astype(int) * 1
        df_enhanced['lifestyle_risk_score'] = lifestyle_risk
        
        # Activity level
        if 'steps_per_day' in df_enhanced.columns:
            df_enhanced['activity_level'] = pd.cut(df_enhanced['steps_per_day'], 
                                                  bins=[0, 5000, 8000, 12000, 50000], 
                                                  labels=['sedentary', 'low', 'moderate', 'high'])
        
        # Service type risk (based on historical data)
        service_risk_map = {
            'Surgery': 3,
            'Therapy': 2,
            'Consultation': 1,
            'Medication': 1
        }
        if 'service_type' in df_enhanced.columns:
            df_enhanced['service_risk'] = df_enhanced['service_type'].map(service_risk_map).fillna(1)
        
        # Cost-based risk (high cost procedures may indicate complexity)
        if 'cost_of_service' in df_enhanced.columns:
            df_enhanced['cost_risk_category'] = pd.qcut(df_enhanced['cost_of_service'], 
                                                       q=4, labels=['low', 'medium', 'high', 'very_high'])
        
        # Interaction features
        if 'age' in df_enhanced.columns and 'comorbidity_count' in df_enhanced.columns:
            df_enhanced['age_comorbidity_interaction'] = df_enhanced['age'] * df_enhanced['comorbidity_count']
        
        return df_enhanced
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Define comprehensive feature set
        numerical_features = [
            'age', 'rbc_count', 'wbc_count', 'systole', 'diastole', 'heart_rate',
            'blood_sugar', 'bmi', 'cholesterol', 'steps_per_day', 'calorie_intake',
            'cost_of_service', 'comorbidity_count', 'lifestyle_risk_score',
            'service_risk', 'age_comorbidity_interaction'
        ]
        
        categorical_features = [
            'gender', 'allergies', 'smoking', 'alcohol', 'sleep_pattern',
            'service_type', 'diagnosis_code', 'procedure_code',
            'age_risk_category', 'bmi_category', 'bp_category',
            'activity_level', 'cost_risk_category'
        ]
        
        # Select available features
        available_numerical = [f for f in numerical_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        # Prepare numerical features
        X_numerical = df[available_numerical].fillna(df[available_numerical].median())
        
        # Prepare categorical features
        X_categorical = pd.DataFrame()
        for feature in available_categorical:
            if feature in df.columns:
                # Use label encoding for high cardinality features
                if feature in ['diagnosis_code', 'procedure_code']:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                        X_categorical[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                    else:
                        # Handle unseen categories
                        try:
                            X_categorical[feature] = self.label_encoders[feature].transform(df[feature].astype(str))
                        except ValueError:
                            # For unseen categories, assign a default value
                            X_categorical[feature] = 0
                else:
                    # One-hot encoding for low cardinality features
                    dummies = pd.get_dummies(df[feature], prefix=feature)
                    X_categorical = pd.concat([X_categorical, dummies], axis=1)
        
        # Combine features
        X = pd.concat([X_numerical, X_categorical], axis=1)
        self.feature_names = X.columns.tolist()
        
        return X
    
    def train_model(self, df, target_column='readmission_30d'):
        """Train enhanced readmission prediction model"""
        print("Starting enhanced readmission model training...")
        
        # Engineer features
        df_enhanced = self.engineer_features(df)
        
        # Prepare features
        X = self.prepare_features(df_enhanced)
        y = df_enhanced[target_column]
        
        print(f"Training with {X.shape[1]} features on {X.shape[0]} samples")
        print(f"Readmission rate: {y.mean():.2%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Define models for ensemble
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'xgb': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'gb': GradientBoostingClassifier(random_state=42),
            'lr': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        }
        
        # Train individual models and create ensemble
        trained_models = []
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_selected, y_train)
            trained_models.append((name, model))
        
        # Create voting classifier
        self.model = VotingClassifier(trained_models, voting='soft')
        self.model.fit(X_train_selected, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_selected)
        y_pred_proba = self.model.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'cross_val_accuracy': cross_val_score(self.model, X_train_selected, y_train, cv=5).mean()
        }
        
        print("\nModel Performance:")
        for metric, value in self.performance_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Feature importance
        if hasattr(self.model.estimators_[0], 'feature_importances_'):
            selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
            feature_importance = self.model.estimators_[0].feature_importances_
            self.feature_importance = dict(zip(selected_features, feature_importance))
        
        return self.performance_metrics
    
    def predict(self, df):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Engineer features
        df_enhanced = self.engineer_features(df)
        
        # Prepare features
        X = self.prepare_features(df_enhanced)
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_selected)
        probabilities = self.model.predict_proba(X_selected)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        print(f"Model loaded from {filepath}")

def train_enhanced_model():
    """Train the enhanced readmission prediction model"""
    # Load data
    claim_df = pd.read_csv('newdatasets/claim_new.csv')
    merged_df = pd.read_csv('data/merged.csv')
    
    # Merge datasets
    full_df = pd.merge(claim_df, merged_df, on=['claim_id', 'patient_id'], how='inner')
    
    # Initialize and train model
    predictor = EnhancedReadmissionPredictor()
    performance = predictor.train_model(full_df)
    
    # Save model
    predictor.save_model('enhanced_readmission_model.pkl')
    
    return predictor, performance

if __name__ == "__main__":
    predictor, performance = train_enhanced_model()
    print(f"\nFinal Model Accuracy: {performance['accuracy']:.2%}")
    if performance['accuracy'] >= 0.8:
        print("✅ Target accuracy of 80% achieved!")
    else:
        print("❌ Target accuracy of 80% not yet achieved. Consider additional feature engineering.")
