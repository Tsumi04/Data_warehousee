"""
Machine Learning Analytics Engine for Retail Data Warehouse
This module provides advanced ML capabilities for predictive analytics,
customer segmentation, demand forecasting, and anomaly detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging
import joblib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Deep learning features will be disabled.")

from config import DATABASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerSegmentationML:
    """Advanced customer segmentation using ML"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_model = None
        self.is_trained = False
        self.segment_labels = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for customer segmentation"""
        logger.info("Preparing features for customer segmentation...")
        
        # RFM Features
        features_df = df.groupby('CustomerKey').agg({
            'InvoiceDate': ['min', 'max', 'count'],
            'TotalRevenue': ['sum', 'mean'],
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        # Flatten column names
        features_df.columns = ['CustomerKey', 'FirstPurchase', 'LastPurchase', 
                              'PurchaseFrequency', 'TotalRevenue', 'AvgOrderValue', 
                              'TotalQuantity', 'UniqueOrders']
        
        # Calculate RFM metrics
        current_date = datetime.now()
        features_df['Recency'] = (current_date - pd.to_datetime(features_df['LastPurchase'])).dt.days
        features_df['Frequency'] = features_df['PurchaseFrequency']
        features_df['Monetary'] = features_df['TotalRevenue']
        
        # Additional features
        features_df['AvgOrderSize'] = features_df['TotalQuantity'] / features_df['UniqueOrders']
        features_df['RevenuePerOrder'] = features_df['TotalRevenue'] / features_df['UniqueOrders']
        features_df['CustomerLifespan'] = (pd.to_datetime(features_df['LastPurchase']) - 
                                          pd.to_datetime(features_df['FirstPurchase'])).dt.days
        
        # Handle infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        # Select features for clustering
        feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderSize', 
                          'RevenuePerOrder', 'CustomerLifespan']
        
        return features_df[['CustomerKey'] + feature_columns]
    
    def train_kmeans_segmentation(self, features_df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """Train K-Means clustering model"""
        logger.info(f"Training K-Means segmentation with {n_clusters} clusters...")
        
        feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderSize', 
                          'RevenuePerOrder', 'CustomerLifespan']
        
        X = features_df[feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-Means
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        features_df['KMeans_Segment'] = cluster_labels
        
        # Analyze segments
        segment_analysis = self._analyze_segments(features_df, 'KMeans_Segment')
        
        self.is_trained = True
        logger.info("K-Means segmentation completed successfully")
        
        return {
            'model': self.kmeans_model,
            'scaler': self.scaler,
            'segments': segment_analysis,
            'feature_columns': feature_columns
        }
    
    def train_dbscan_segmentation(self, features_df: pd.DataFrame, 
                                 eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """Train DBSCAN clustering model"""
        logger.info("Training DBSCAN segmentation...")
        
        feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderSize', 
                          'RevenuePerOrder', 'CustomerLifespan']
        
        X = features_df[feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train DBSCAN
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.dbscan_model.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        features_df['DBSCAN_Segment'] = cluster_labels
        
        # Analyze segments
        segment_analysis = self._analyze_segments(features_df, 'DBSCAN_Segment')
        
        logger.info("DBSCAN segmentation completed successfully")
        
        return {
            'model': self.dbscan_model,
            'scaler': self.scaler,
            'segments': segment_analysis,
            'feature_columns': feature_columns
        }
    
    def _analyze_segments(self, df: pd.DataFrame, segment_column: str) -> Dict[str, Any]:
        """Analyze customer segments"""
        segment_stats = df.groupby(segment_column).agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'AvgOrderSize': 'mean',
            'RevenuePerOrder': 'mean',
            'CustomerLifespan': 'mean',
            'CustomerKey': 'count'
        }).round(2)
        
        segment_stats.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 
                               'Avg_OrderSize', 'Avg_RevenuePerOrder', 'Avg_Lifespan', 'Customer_Count']
        
        # Create segment descriptions
        segment_descriptions = {}
        for segment in segment_stats.index:
            if segment == -1:  # DBSCAN noise
                segment_descriptions[segment] = "Noise/Outliers"
            else:
                recency = segment_stats.loc[segment, 'Avg_Recency']
                frequency = segment_stats.loc[segment, 'Avg_Frequency']
                monetary = segment_stats.loc[segment, 'Avg_Monetary']
                
                if recency < 30 and frequency > 10 and monetary > 1000:
                    segment_descriptions[segment] = "Champions"
                elif recency < 60 and frequency > 5 and monetary > 500:
                    segment_descriptions[segment] = "Loyal Customers"
                elif recency < 30 and frequency < 5 and monetary > 500:
                    segment_descriptions[segment] = "Potential Loyalists"
                elif recency > 90 and frequency > 5 and monetary > 500:
                    segment_descriptions[segment] = "At Risk"
                else:
                    segment_descriptions[segment] = f"Segment {segment}"
        
        return {
            'statistics': segment_stats.to_dict('index'),
            'descriptions': segment_descriptions
        }
    
    def predict_segment(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Predict customer segments for new data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderSize', 
                          'RevenuePerOrder', 'CustomerLifespan']
        
        X = customer_data[feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        if self.kmeans_model:
            predictions = self.kmeans_model.predict(X_scaled)
            customer_data['Predicted_Segment'] = predictions
        
        return customer_data

class ChurnPredictionML:
    """Customer churn prediction using ML"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None
    
    def prepare_churn_features(self, df: pd.DataFrame, churn_threshold_days: int = 90) -> pd.DataFrame:
        """Prepare features for churn prediction"""
        logger.info("Preparing features for churn prediction...")
        
        # Calculate churn status
        current_date = datetime.now()
        df['DaysSinceLastPurchase'] = (current_date - pd.to_datetime(df['LastPurchaseDate'])).dt.days
        df['IsChurned'] = (df['DaysSinceLastPurchase'] > churn_threshold_days).astype(int)
        
        # Create features
        features_df = df.groupby('CustomerKey').agg({
            'TotalRevenue': ['sum', 'mean', 'std'],
            'TotalOrders': ['sum', 'mean'],
            'CustomerRecency': 'mean',
            'CustomerFrequency': 'mean',
            'CustomerMonetary': 'mean',
            'DaysSinceLastPurchase': 'mean',
            'IsChurned': 'first'
        }).reset_index()
        
        # Flatten column names
        features_df.columns = ['CustomerKey', 'TotalRevenue_Sum', 'TotalRevenue_Mean', 'TotalRevenue_Std',
                              'TotalOrders_Sum', 'TotalOrders_Mean', 'AvgRecency', 'AvgFrequency',
                              'AvgMonetary', 'DaysSinceLastPurchase', 'IsChurned']
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Add derived features
        features_df['RevenueTrend'] = features_df['TotalRevenue_Mean'] / (features_df['TotalRevenue_Std'] + 1)
        features_df['OrderFrequency'] = features_df['TotalOrders_Sum'] / (features_df['AvgRecency'] + 1)
        features_df['ValuePerOrder'] = features_df['TotalRevenue_Sum'] / (features_df['TotalOrders_Sum'] + 1)
        
        return features_df
    
    def train_churn_model(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train churn prediction model"""
        logger.info("Training churn prediction model...")
        
        # Prepare features and target
        feature_columns = ['TotalRevenue_Sum', 'TotalRevenue_Mean', 'TotalRevenue_Std',
                          'TotalOrders_Sum', 'TotalOrders_Mean', 'AvgRecency', 'AvgFrequency',
                          'AvgMonetary', 'DaysSinceLastPurchase', 'RevenueTrend', 
                          'OrderFrequency', 'ValuePerOrder']
        
        X = features_df[feature_columns].values
        y = features_df['IsChurned'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = self.model.score(X_test, y_test)
        feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
        
        self.is_trained = True
        self.feature_importance = feature_importance
        
        logger.info(f"Churn prediction model trained - Accuracy: {accuracy:.3f}")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'accuracy': accuracy,
            'cv_scores': cv_scores.tolist(),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict_churn(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Predict churn probability for customers"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        feature_columns = ['TotalRevenue_Sum', 'TotalRevenue_Mean', 'TotalRevenue_Std',
                          'TotalOrders_Sum', 'TotalOrders_Mean', 'AvgRecency', 'AvgFrequency',
                          'AvgMonetary', 'DaysSinceLastPurchase', 'RevenueTrend', 
                          'OrderFrequency', 'ValuePerOrder']
        
        X = customer_data[feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        churn_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        churn_predictions = self.model.predict(X_scaled)
        
        customer_data['ChurnProbability'] = churn_probabilities
        customer_data['PredictedChurn'] = churn_predictions
        customer_data['ChurnRisk'] = pd.cut(churn_probabilities, 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
        
        return customer_data

class DemandForecastingML:
    """Demand forecasting using time series analysis"""
    
    def __init__(self):
        self.models = {}
        self.is_trained = False
    
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                frequency: str = 'D') -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        logger.info("Preparing time series data for demand forecasting...")
        
        # Aggregate data by date
        ts_data = df.groupby('InvoiceDate').agg({
            'TotalRevenue': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerKey': 'nunique'
        }).reset_index()
        
        ts_data.columns = ['Date', 'Revenue', 'Quantity', 'Orders', 'Customers']
        ts_data['Date'] = pd.to_datetime(ts_data['Date'])
        ts_data = ts_data.set_index('Date')
        
        # Resample to desired frequency
        ts_data = ts_data.resample(frequency).sum()
        
        # Fill missing values
        ts_data = ts_data.fillna(0)
        
        return ts_data
    
    def train_arima_model(self, ts_data: pd.DataFrame, 
                         target_column: str = 'Revenue',
                         order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """Train ARIMA model for time series forecasting"""
        logger.info(f"Training ARIMA model for {target_column}...")
        
        series = ts_data[target_column]
        
        # Fit ARIMA model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Store model
        self.models[f'{target_column}_ARIMA'] = fitted_model
        
        # Calculate metrics
        predictions = fitted_model.fittedvalues
        mse = mean_squared_error(series, predictions)
        rmse = np.sqrt(mse)
        
        logger.info(f"ARIMA model trained - RMSE: {rmse:.2f}")
        
        return {
            'model': fitted_model,
            'rmse': rmse,
            'mse': mse,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
    
    def train_exponential_smoothing(self, ts_data: pd.DataFrame,
                                   target_column: str = 'Revenue') -> Dict[str, Any]:
        """Train Exponential Smoothing model"""
        logger.info(f"Training Exponential Smoothing model for {target_column}...")
        
        series = ts_data[target_column]
        
        # Fit Exponential Smoothing
        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
        fitted_model = model.fit()
        
        # Store model
        self.models[f'{target_column}_ES'] = fitted_model
        
        # Calculate metrics
        predictions = fitted_model.fittedvalues
        mse = mean_squared_error(series, predictions)
        rmse = np.sqrt(mse)
        
        logger.info(f"Exponential Smoothing model trained - RMSE: {rmse:.2f}")
        
        return {
            'model': fitted_model,
            'rmse': rmse,
            'mse': mse,
            'aic': fitted_model.aic
        }
    
    def forecast_demand(self, target_column: str, periods: int = 30) -> Dict[str, Any]:
        """Generate demand forecasts"""
        if not self.models:
            raise ValueError("No models trained yet. Please train models first.")
        
        forecasts = {}
        
        for model_name, model in self.models.items():
            if target_column in model_name:
                # Generate forecast
                forecast = model.forecast(steps=periods)
                confidence_intervals = model.get_forecast(steps=periods).conf_int()
                
                forecasts[model_name] = {
                    'forecast': forecast.tolist(),
                    'confidence_intervals': confidence_intervals.values.tolist(),
                    'forecast_dates': pd.date_range(
                        start=datetime.now(), 
                        periods=periods, 
                        freq='D'
                    ).strftime('%Y-%m-%d').tolist()
                }
        
        return forecasts

class AnomalyDetectionML:
    """Advanced anomaly detection using ML"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
    
    def prepare_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection"""
        logger.info("Preparing features for anomaly detection...")
        
        # Create features for anomaly detection
        features_df = df.groupby('InvoiceNo').agg({
            'TotalRevenue': ['sum', 'mean', 'std'],
            'Quantity': ['sum', 'mean', 'std'],
            'UnitPrice': ['mean', 'std'],
            'CustomerKey': 'nunique',
            'ProductKey': 'nunique'
        }).reset_index()
        
        # Flatten column names
        features_df.columns = ['InvoiceNo', 'TotalRevenue_Sum', 'TotalRevenue_Mean', 'TotalRevenue_Std',
                              'Quantity_Sum', 'Quantity_Mean', 'Quantity_Std',
                              'UnitPrice_Mean', 'UnitPrice_Std',
                              'UniqueCustomers', 'UniqueProducts']
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Add derived features
        features_df['RevenuePerItem'] = features_df['TotalRevenue_Sum'] / (features_df['Quantity_Sum'] + 1)
        features_df['ItemsPerCustomer'] = features_df['Quantity_Sum'] / (features_df['UniqueCustomers'] + 1)
        features_df['ProductsPerOrder'] = features_df['UniqueProducts']
        
        return features_df
    
    def train_isolation_forest(self, features_df: pd.DataFrame,
                              contamination: float = 0.1) -> Dict[str, Any]:
        """Train Isolation Forest for anomaly detection"""
        logger.info("Training Isolation Forest for anomaly detection...")
        
        feature_columns = ['TotalRevenue_Sum', 'TotalRevenue_Mean', 'TotalRevenue_Std',
                          'Quantity_Sum', 'Quantity_Mean', 'Quantity_Std',
                          'UnitPrice_Mean', 'UnitPrice_Std',
                          'UniqueCustomers', 'UniqueProducts',
                          'RevenuePerItem', 'ItemsPerCustomer', 'ProductsPerOrder']
        
        X = features_df[feature_columns].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_scaled)
        
        # Store model and scaler
        self.models['IsolationForest'] = model
        self.scalers['IsolationForest'] = scaler
        
        # Predict anomalies
        anomaly_scores = model.score_samples(X_scaled)
        anomaly_predictions = model.predict(X_scaled)
        
        # Add results to dataframe
        features_df['AnomalyScore'] = anomaly_scores
        features_df['IsAnomaly'] = anomaly_predictions == -1
        
        self.is_trained = True
        
        anomaly_count = (anomaly_predictions == -1).sum()
        logger.info(f"Isolation Forest trained - Found {anomaly_count} anomalies")
        
        return {
            'model': model,
            'scaler': scaler,
            'anomaly_count': int(anomaly_count),
            'anomaly_percentage': float(anomaly_count / len(features_df) * 100),
            'feature_columns': feature_columns
        }
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in new data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        feature_columns = ['TotalRevenue_Sum', 'TotalRevenue_Mean', 'TotalRevenue_Std',
                          'Quantity_Sum', 'Quantity_Mean', 'Quantity_Std',
                          'UnitPrice_Mean', 'UnitPrice_Std',
                          'UniqueCustomers', 'UniqueProducts',
                          'RevenuePerItem', 'ItemsPerCustomer', 'ProductsPerOrder']
        
        X = data[feature_columns].values
        X_scaled = self.scalers['IsolationForest'].transform(X)
        
        # Predict anomalies
        anomaly_scores = self.models['IsolationForest'].score_samples(X_scaled)
        anomaly_predictions = self.models['IsolationForest'].predict(X_scaled)
        
        data['AnomalyScore'] = anomaly_scores
        data['IsAnomaly'] = anomaly_predictions == -1
        
        return data

class MLAnalyticsEngine:
    """Main ML Analytics Engine that orchestrates all ML models"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url)
        self.customer_segmentation = CustomerSegmentationML()
        self.churn_prediction = ChurnPredictionML()
        self.demand_forecasting = DemandForecastingML()
        self.anomaly_detection = AnomalyDetectionML()
        
        # Model storage
        self.models_dir = Path("ml_models")
        self.models_dir.mkdir(exist_ok=True)
    
    def load_data_for_ml(self) -> Dict[str, pd.DataFrame]:
        """Load data from data warehouse for ML analysis"""
        logger.info("Loading data from data warehouse for ML analysis...")
        
        # Load customer data
        customer_query = """
        SELECT 
            c.CustomerKey,
            c.CustomerID,
            c.Country,
            c.CustomerSegment,
            c.CustomerLifetimeValue,
            c.CustomerRecency,
            c.CustomerFrequency,
            c.CustomerMonetary,
            c.FirstPurchaseDate,
            c.LastPurchaseDate,
            c.TotalOrders,
            c.TotalRevenue,
            c.AverageOrderValue
        FROM DimCustomer c
        WHERE c.CustomerID != 'UNKNOWN'
        """
        
        customer_df = pd.read_sql(customer_query, self.engine)
        
        # Load sales data
        sales_query = """
        SELECT 
            fs.InvoiceNo,
            fs.InvoiceDate,
            fs.CustomerKey,
            fs.ProductKey,
            fs.Quantity,
            fs.UnitPrice,
            fs.TotalRevenue,
            d.FullDate as InvoiceDate
        FROM FactSales fs
        JOIN DimDate d ON fs.DateKey = d.DateKey
        """
        
        sales_df = pd.read_sql(sales_query, self.engine)
        sales_df['InvoiceDate'] = pd.to_datetime(sales_df['InvoiceDate'])
        
        logger.info(f"Loaded {len(customer_df):,} customers and {len(sales_df):,} sales records")
        
        return {
            'customers': customer_df,
            'sales': sales_df
        }
    
    def run_customer_segmentation_analysis(self) -> Dict[str, Any]:
        """Run complete customer segmentation analysis"""
        logger.info("Running customer segmentation analysis...")
        
        # Load data
        data = self.load_data_for_ml()
        customer_df = data['customers']
        sales_df = data['sales']
        
        # Merge customer and sales data
        merged_df = sales_df.merge(customer_df, on='CustomerKey', how='inner')
        
        # Prepare features
        features_df = self.customer_segmentation.prepare_features(merged_df)
        
        # Train K-Means
        kmeans_results = self.customer_segmentation.train_kmeans_segmentation(features_df, n_clusters=5)
        
        # Train DBSCAN
        dbscan_results = self.customer_segmentation.train_dbscan_segmentation(features_df)
        
        # Save models
        self._save_models('customer_segmentation', {
            'kmeans': kmeans_results,
            'dbscan': dbscan_results
        })
        
        return {
            'kmeans_segments': kmeans_results['segments'],
            'dbscan_segments': dbscan_results['segments'],
            'feature_analysis': features_df.describe().to_dict()
        }
    
    def run_churn_prediction_analysis(self) -> Dict[str, Any]:
        """Run churn prediction analysis"""
        logger.info("Running churn prediction analysis...")
        
        # Load data
        data = self.load_data_for_ml()
        customer_df = data['customers']
        
        # Prepare features
        features_df = self.churn_prediction.prepare_churn_features(customer_df)
        
        # Train model
        model_results = self.churn_prediction.train_churn_model(features_df)
        
        # Save models
        self._save_models('churn_prediction', model_results)
        
        return {
            'model_performance': {
                'accuracy': model_results['accuracy'],
                'cv_scores': model_results['cv_scores']
            },
            'feature_importance': model_results['feature_importance'],
            'classification_report': model_results['classification_report']
        }
    
    def run_demand_forecasting_analysis(self) -> Dict[str, Any]:
        """Run demand forecasting analysis"""
        logger.info("Running demand forecasting analysis...")
        
        # Load data
        data = self.load_data_for_ml()
        sales_df = data['sales']
        
        # Prepare time series data
        ts_data = self.demand_forecasting.prepare_time_series_data(sales_df)
        
        # Train models
        arima_results = self.demand_forecasting.train_arima_model(ts_data, 'Revenue')
        es_results = self.demand_forecasting.train_exponential_smoothing(ts_data, 'Revenue')
        
        # Generate forecasts
        forecasts = self.demand_forecasting.forecast_demand('Revenue', periods=30)
        
        # Save models
        self._save_models('demand_forecasting', {
            'arima': arima_results,
            'exponential_smoothing': es_results,
            'forecasts': forecasts
        })
        
        return {
            'arima_performance': {
                'rmse': arima_results['rmse'],
                'aic': arima_results['aic']
            },
            'es_performance': {
                'rmse': es_results['rmse'],
                'aic': es_results['aic']
            },
            'forecasts': forecasts
        }
    
    def run_anomaly_detection_analysis(self) -> Dict[str, Any]:
        """Run anomaly detection analysis"""
        logger.info("Running anomaly detection analysis...")
        
        # Load data
        data = self.load_data_for_ml()
        sales_df = data['sales']
        
        # Prepare features
        features_df = self.anomaly_detection.prepare_anomaly_features(sales_df)
        
        # Train model
        model_results = self.anomaly_detection.train_isolation_forest(features_df)
        
        # Save models
        self._save_models('anomaly_detection', model_results)
        
        return {
            'anomaly_statistics': {
                'total_anomalies': model_results['anomaly_count'],
                'anomaly_percentage': model_results['anomaly_percentage']
            },
            'feature_columns': model_results['feature_columns']
        }
    
    def run_complete_ml_analysis(self) -> Dict[str, Any]:
        """Run complete ML analysis suite with enhanced error handling"""
        logger.info("Running complete ML analysis suite...")
        
        results = {}
        
        # Check if we have enough data first
        try:
            from sqlalchemy import create_engine, text
            from config import DATABASE_URL
            
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM FactSales"))
                total_records = result.fetchone()[0]
                
                if total_records < 1000:
                    logger.warning(f"Not enough data for ML analysis: {total_records} records")
                    return {'error': f'Insufficient data: {total_records} records (minimum 1000 required)'}
                
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return {'error': f'Database connection error: {str(e)}'}
        
        try:
            # Customer Segmentation
            logger.info("Running customer segmentation analysis...")
            results['customer_segmentation'] = self.run_customer_segmentation_analysis()
        except Exception as e:
            logger.error(f"Error in customer segmentation: {e}")
            results['customer_segmentation'] = {'error': str(e), 'status': 'failed'}
        
        try:
            # Churn Prediction
            logger.info("Running churn prediction analysis...")
            results['churn_prediction'] = self.run_churn_prediction_analysis()
        except Exception as e:
            logger.error(f"Error in churn prediction: {e}")
            results['churn_prediction'] = {'error': str(e), 'status': 'failed'}
        
        try:
            # Demand Forecasting
            logger.info("Running demand forecasting analysis...")
            results['demand_forecasting'] = self.run_demand_forecasting_analysis()
        except Exception as e:
            logger.error(f"Error in demand forecasting: {e}")
            results['demand_forecasting'] = {'error': str(e), 'status': 'failed'}
        
        try:
            # Anomaly Detection
            logger.info("Running anomaly detection analysis...")
            results['anomaly_detection'] = self.run_anomaly_detection_analysis()
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            results['anomaly_detection'] = {'error': str(e), 'status': 'failed'}
        
        # Save complete results
        try:
            self._save_analysis_results(results)
        except Exception as e:
            logger.warning(f"Error saving analysis results: {e}")
        
        # Count successful analyses
        successful_analyses = sum(1 for result in results.values() if 'error' not in result)
        total_analyses = len(results)
        
        logger.info(f"Complete ML analysis suite completed: {successful_analyses}/{total_analyses} successful")
        return results
    
    def _save_models(self, model_type: str, models: Dict[str, Any]):
        """Save ML models to disk"""
        model_file = self.models_dir / f"{model_type}_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(models, f)
        
        logger.info(f"Models saved to: {model_file}")
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to JSON"""
        results_file = self.models_dir / f"ml_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to: {results_file}")

def main():
    """Main function to run ML analytics"""
    print("🤖 Starting ML Analytics Engine...")
    
    # Initialize ML engine
    ml_engine = MLAnalyticsEngine()
    
    # Run complete analysis
    results = ml_engine.run_complete_ml_analysis()
    
    # Print summary
    print("\n📊 ML Analysis Results Summary:")
    print("=" * 50)
    
    for analysis_type, result in results.items():
        if 'error' in result:
            print(f"❌ {analysis_type}: {result['error']}")
        else:
            print(f"✅ {analysis_type}: Completed successfully")
    
    print("\n🎉 ML Analytics Engine completed!")

if __name__ == '__main__':
    main()
