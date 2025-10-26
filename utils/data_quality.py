"""
Data Quality Module
Contains functions for data validation and quality checks.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_quantity(quantity, min_value=1, max_value=100000):
    """
    Validate product quantity.
    
    Args:
        quantity: Quantity value to validate
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        qty = float(quantity)
        return min_value <= qty <= max_value and qty > 0
    except (ValueError, TypeError):
        return False


def validate_price(price, min_value=0.01, max_value=50000):
    """
    Validate unit price.
    
    Args:
        price: Price value to validate
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        p = float(price)
        return min_value <= p <= max_value and p > 0
    except (ValueError, TypeError):
        return False


def detect_outliers_iqr(df, column):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: pandas.DataFrame
        column: Column name to check
    
    Returns:
        pandas.DataFrame with outlier information
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    logger.info(f"Found {len(outliers)} outliers in {column}")
    logger.info(f"Range: {lower_bound:.2f} to {upper_bound:.2f}")
    
    return outliers


def calculate_data_quality_score(df):
    """
    Calculate overall data quality score.
    
    Args:
        df: pandas.DataFrame
    
    Returns:
        dict: Data quality metrics
    """
    total_rows = len(df)
    
    quality_metrics = {
        'total_rows': total_rows,
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (total_rows * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'completeness_score': ((total_rows - df.isnull().any(axis=1).sum()) / total_rows) * 100
    }
    
    logger.info("Data Quality Metrics:")
    for key, value in quality_metrics.items():
        logger.info(f"  {key}: {value}")
    
    return quality_metrics


def clean_text(text):
    """
    Clean text data.
    
    Args:
        text: String to clean
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return None
    
    text = str(text).strip()
    
    if text == '' or text.lower() == 'nan':
        return None
    
    return text


