"""
Advanced ETL (Extract, Transform, Load) Pipeline for Retail Data Warehouse
This module implements a production-ready ETL pipeline with comprehensive
data quality checks, monitoring, and error handling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging
import json
import hashlib
from typing import Dict, List, Tuple, Any
from pathlib import Path

from config import (
    RAW_DATA_PATH, DATABASE_URL, 
    MIN_QUANTITY, MAX_QUANTITY, MIN_UNIT_PRICE, MAX_UNIT_PRICE,
    BATCH_SIZE
)
from data_quality_framework import DataQualityFramework, DataProfiler
from advanced_dwh_design import Base, DimDate, DimCustomer, DimProduct, DimLocation, FactSales

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl.log'),
        logging.StreamHandler()
    ]
)

# Setup detailed logging handler
detailed_handler = logging.FileHandler('etl_detailed.log')
detailed_handler.setLevel(logging.DEBUG)
detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
detailed_handler.setFormatter(detailed_formatter)
logger = logging.getLogger(__name__)
logger.addHandler(detailed_handler)

# ETL Run ID for tracking
ETL_RUN_ID = f"ETL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logger.info(f"Starting ETL Run: {ETL_RUN_ID}")


# ==================== EXTRACT PHASE ====================

def extract_data(file_path=None, chunk_size=None):
    """
    Enhanced data extraction with chunking and profiling.
    
    Args:
        file_path: Path to the source data file. If None, uses config default.
        chunk_size: Size of chunks for processing large files
    
    Returns:
        pandas.DataFrame: Raw data extracted from source
    """
    if file_path is None:
        file_path = RAW_DATA_PATH
    
    if chunk_size is None:
        chunk_size = 50000  # Default chunk size
    
    logger.info(f"Starting EXTRACT phase...")
    logger.info(f"Reading data from: {file_path}")
    logger.info(f"Chunk size: {chunk_size:,} records")
    
    try:
        # Check file size
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Source file size: {file_size:.2f} MB")
        
        # Read CSV with appropriate encoding and chunking for large files
        if file_size > 100:  # If file is larger than 100MB, use chunking
            logger.info("Large file detected, using chunked reading...")
            chunks = []
            for chunk in pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        # Remove index column if present (added by pandas during export)
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        # Generate data profile
        profiler = DataProfiler()
        profile = profiler.profile_dataframe(df, 'RawData')
        
        # Log data profile
        logger.info(f"Data Profile - Records: {profile['total_records']:,}, Columns: {profile['total_columns']}")
        logger.info(f"Memory usage: {profile['memory_usage'] / (1024*1024):.2f} MB")
        
        # Log column statistics
        for col, stats in profile['columns'].items():
            logger.debug(f"Column {col}: {stats['data_type']}, Nulls: {stats['null_count']} ({stats['null_percentage']:.1f}%)")
        
        # Save profile to file
        with open(f'data_profile_{ETL_RUN_ID}.json', 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        logger.info(f"Successfully extracted {len(df):,} records from source")
        logger.info(f"Initial columns: {df.columns.tolist()}")
        
        return df
    
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        logger.error("Please download the dataset from: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
        raise
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise


# ==================== TRANSFORM PHASE ====================

def transform_data(df):
    """
    Enhanced data transformation with comprehensive data quality checks.
    This is the most complex phase, containing all business logic.
    
    Args:
        df: pandas.DataFrame with raw data
    
    Returns:
        tuple: (transformed_fact_df, dim_date_df, dim_customer_df, dim_product_df, dim_location_df, quality_results)
    """
    logger.info("Starting TRANSFORM phase...")
    logger.info(f"Input data shape: {df.shape}")
    
    # Initialize data quality framework
    dq_framework = DataQualityFramework(ETL_RUN_ID)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Run initial data quality checks on raw data
    logger.info("Running initial data quality checks...")
    raw_quality_results = dq_framework.run_all_checks(df, 'RawData')
    
    # Log quality issues
    critical_issues = [r for r in raw_quality_results if r.severity.value == 'critical']
    if critical_issues:
        logger.warning(f"Found {len(critical_issues)} critical data quality issues in raw data")
        for issue in critical_issues:
            logger.warning(f"  - {issue.check_name}: {issue.error_message}")
    
    # Check if we should proceed
    if not dq_framework.should_proceed_with_load():
        logger.error("Critical data quality issues found. Aborting ETL process.")
        raise ValueError("Data quality check failed. Please review and fix data quality issues.")
    
    # ===== STEP 1: Enhanced Data Type Conversion =====
    logger.info("Step 1: Converting data types with validation...")
    
    # Convert InvoiceDate to datetime with error handling
    if 'InvoiceDate' in df.columns:
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            invalid_dates = df['InvoiceDate'].isnull().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates, will be filtered out")
            logger.info("Converted InvoiceDate to datetime")
        except Exception as e:
            logger.error(f"Error converting InvoiceDate: {e}")
            raise
    
    # Convert CustomerID to string with enhanced handling
    if 'CustomerID' in df.columns:
        # Handle different types of missing values
        df['CustomerID'] = df['CustomerID'].fillna('UNKNOWN')
        df['CustomerID'] = df['CustomerID'].astype(str)
        
        # Clean up any 'nan' strings
        df['CustomerID'] = df['CustomerID'].replace('nan', 'UNKNOWN')
        logger.info("Converted CustomerID to string")
    
    # Convert numeric columns with validation
    for col in ['Quantity', 'UnitPrice']:
        if col in df.columns:
            original_count = len(df)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            invalid_count = df[col].isnull().sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid values in {col}, will be filtered out")
    
    # Convert StockCode to string
    if 'StockCode' in df.columns:
        df['StockCode'] = df['StockCode'].astype(str).str.strip()
        logger.info("Converted StockCode to string")
    
    # Convert Description to string and clean
    if 'Description' in df.columns:
        df['Description'] = df['Description'].astype(str).str.strip()
        df['Description'] = df['Description'].replace('nan', '')
        logger.info("Converted Description to string")
    
    # Convert Country to string
    if 'Country' in df.columns:
        df['Country'] = df['Country'].astype(str).str.strip()
        logger.info("Converted Country to string")
    
    # ===== STEP 2: Enhanced Data Quality Checks and Filtering =====
    logger.info("Step 2: Performing comprehensive data quality checks...")
    
    initial_count = len(df)
    logger.info(f"Starting with {initial_count:,} records")
    
    # Track filtering statistics
    filtering_stats = {}
    
    # Remove cancelled orders (InvoiceNo starts with 'C')
    cancelled_mask = df['InvoiceNo'].astype(str).str.startswith('C', na=False)
    cancelled_count = cancelled_mask.sum()
    df = df[~cancelled_mask]
    filtering_stats['cancelled_orders'] = cancelled_count
    logger.info(f"Removed {cancelled_count:,} cancelled orders. Remaining: {len(df):,}")
    
    # Remove invalid quantities
    invalid_qty_mask = (df['Quantity'] <= 0) | (df['Quantity'] > MAX_QUANTITY) | df['Quantity'].isnull()
    invalid_qty_count = invalid_qty_mask.sum()
    df = df[~invalid_qty_mask]
    filtering_stats['invalid_quantities'] = invalid_qty_count
    logger.info(f"Removed {invalid_qty_count:,} records with invalid quantities. Remaining: {len(df):,}")
    
    # Remove invalid prices
    invalid_price_mask = (df['UnitPrice'] <= 0) | (df['UnitPrice'] > MAX_UNIT_PRICE) | df['UnitPrice'].isnull()
    invalid_price_count = invalid_price_mask.sum()
    df = df[~invalid_price_mask]
    filtering_stats['invalid_prices'] = invalid_price_count
    logger.info(f"Removed {invalid_price_count:,} records with invalid prices. Remaining: {len(df):,}")
    
    # Remove rows with missing critical data
    missing_critical_mask = df['InvoiceDate'].isnull() | df['Quantity'].isnull() | df['UnitPrice'].isnull()
    missing_critical_count = missing_critical_mask.sum()
    df = df[~missing_critical_mask]
    filtering_stats['missing_critical_data'] = missing_critical_count
    logger.info(f"Removed {missing_critical_count:,} records with missing critical data. Remaining: {len(df):,}")
    
    # Remove duplicate records (if any)
    duplicate_mask = df.duplicated()
    duplicate_count = duplicate_mask.sum()
    df = df[~duplicate_mask]
    filtering_stats['duplicates'] = duplicate_count
    logger.info(f"Removed {duplicate_count:,} duplicate records. Remaining: {len(df):,}")
    
    # Remove records with empty StockCode or Description
    empty_product_mask = (df['StockCode'].str.strip() == '') | (df['Description'].str.strip() == '')
    empty_product_count = empty_product_mask.sum()
    df = df[~empty_product_mask]
    filtering_stats['empty_product_info'] = empty_product_count
    logger.info(f"Removed {empty_product_count:,} records with empty product info. Remaining: {len(df):,}")
    
    filtered_count = len(df)
    total_filtered = initial_count - filtered_count
    
    logger.info(f"Data quality summary:")
    logger.info(f"  - Initial records: {initial_count:,}")
    logger.info(f"  - Filtered out: {total_filtered:,} ({total_filtered/initial_count*100:.2f}%)")
    logger.info(f"  - Final records: {filtered_count:,} ({filtered_count/initial_count*100:.2f}%)")
    
    # Log detailed filtering statistics
    for filter_type, count in filtering_stats.items():
        logger.info(f"  - {filter_type}: {count:,} records")
    
    # Save filtering statistics
    with open(f'filtering_stats_{ETL_RUN_ID}.json', 'w') as f:
        json.dump(filtering_stats, f, indent=2, default=str)
    
    # ===== STEP 3: Enhanced Enrichment - Calculate Business Measures =====
    logger.info("Step 3: Enriching data with comprehensive business measures...")
    
    # Calculate TotalRevenue with validation
    df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
    
    # Validate revenue calculation
    invalid_revenue = (df['TotalRevenue'] <= 0).sum()
    if invalid_revenue > 0:
        logger.warning(f"Found {invalid_revenue} records with invalid revenue, will be filtered out")
        df = df[df['TotalRevenue'] > 0]
    
    # Calculate additional business measures
    df['LineItemNo'] = 1  # Each row is a line item
    df['OrderQuantity'] = df.groupby('InvoiceNo')['Quantity'].transform('sum')
    
    # Add date components for easier analysis
    df['InvoiceDateDate'] = df['InvoiceDate'].dt.date
    df['InvoiceYear'] = df['InvoiceDate'].dt.year
    df['InvoiceMonth'] = df['InvoiceDate'].dt.month
    df['InvoiceQuarter'] = df['InvoiceDate'].dt.quarter
    df['InvoiceDayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['InvoiceDayOfYear'] = df['InvoiceDate'].dt.dayofyear
    
    # Add business day indicators
    df['IsWeekend'] = df['InvoiceDate'].dt.dayofweek >= 5
    df['IsBusinessDay'] = ~df['IsWeekend']
    
    # Add seasonality
    df['Season'] = df['InvoiceMonth'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Add peak season indicator (December for retail)
    df['IsPeakSeason'] = df['InvoiceMonth'] == 12
    
    logger.info(f"Enhanced data with business measures. Final shape: {df.shape}")
    
    # ===== STEP 4: Build Enhanced Dimension Tables =====
    logger.info("Step 4: Building enhanced dimension tables...")
    
    # Build DimDate with comprehensive time attributes
    dim_date = build_enhanced_dim_date(df['InvoiceDate'].unique())
    logger.info(f"Built DimDate with {len(dim_date)} records")
    
    # Build DimLocation with geographic hierarchy
    dim_location = build_enhanced_dim_location(df)
    logger.info(f"Built DimLocation with {len(dim_location)} records")
    
    # Build DimCustomer with segmentation
    dim_customer = build_enhanced_dim_customer(df)
    logger.info(f"Built DimCustomer with {len(dim_customer)} records")
    
    # Build DimProduct with categorization
    dim_product = build_enhanced_dim_product(df)
    logger.info(f"Built DimProduct with {len(dim_product)} records")
    
    # Build DimChannel (default channel for this dataset)
    dim_channel = build_dim_channel()
    logger.info(f"Built DimChannel with {len(dim_channel)} records")
    
    # ===== STEP 5: Prepare Enhanced Fact Table Data =====
    logger.info("Step 5: Preparing enhanced fact table data...")
    
    # Merge to get surrogate keys with validation
    df['InvoiceDateDate'] = df['InvoiceDate'].dt.date
    
    # Merge Date dimension
    df = df.merge(dim_date[['FullDate', 'DateKey']], 
                  left_on='InvoiceDateDate', 
                  right_on='FullDate', 
                  how='left')
    
    # Merge Customer dimension
    df = df.merge(dim_customer[['CustomerID', 'CustomerKey']], 
                  on='CustomerID', 
                  how='left')
    
    # Merge Product dimension
    df = df.merge(dim_product[['StockCode', 'ProductKey']], 
                  on='StockCode', 
                  how='left')
    
    # Merge Location dimension
    df = df.merge(dim_location[['Country', 'LocationKey']], 
                  on='Country', 
                  how='left')
    
    # Merge Channel dimension (default channel for all records)
    df['ChannelKey'] = 1  # Default channel
    
    # Prepare enhanced fact table
    fact_df = df[[
        'DateKey', 'CustomerKey', 'ProductKey', 'LocationKey', 'ChannelKey',
        'InvoiceNo', 'LineItemNo', 'Quantity', 'UnitPrice', 'TotalRevenue'
    ]].copy()
    
    # Add additional measures
    fact_df['CostAmount'] = 0  # Not available in source data
    fact_df['GrossProfit'] = fact_df['TotalRevenue'] - fact_df['CostAmount']
    fact_df['GrossMargin'] = (fact_df['GrossProfit'] / fact_df['TotalRevenue'] * 100).round(2)
    fact_df['DiscountAmount'] = 0  # Not available in source data
    fact_df['NetRevenue'] = fact_df['TotalRevenue'] - fact_df['DiscountAmount']
    fact_df['TaxAmount'] = 0  # Not available in source data
    fact_df['ReturnQuantity'] = 0  # Not available in source data
    fact_df['ReturnAmount'] = 0  # Not available in source data
    fact_df['DaysToShip'] = 0  # Not available in source data
    fact_df['DaysToDeliver'] = 0  # Not available in source data
    
    # Add ETL metadata
    fact_df['LoadTimestamp'] = datetime.now()
    fact_df['SourceSystem'] = 'Online_Retail'
    fact_df['DataQualityScore'] = 1.0  # Will be calculated based on quality checks
    
    # Validate referential integrity
    foreign_key_cols = ['DateKey', 'CustomerKey', 'ProductKey', 'LocationKey', 'ChannelKey']
    missing_keys = fact_df[foreign_key_cols].isna().any(axis=1).sum()
    
    if missing_keys > 0:
        logger.warning(f"Found {missing_keys} records with missing foreign keys. Removing...")
        fact_df = fact_df.dropna(subset=foreign_key_cols)
    
    # Run final data quality checks on fact table
    logger.info("Running final data quality checks on fact table...")
    fact_quality_results = dq_framework.run_all_checks(fact_df, 'FactSales', ['SalesKey'])
    
    # Calculate data quality score
    if fact_quality_results:
        total_issues = len(fact_quality_results)
        critical_issues = len([r for r in fact_quality_results if r.severity.value == 'critical'])
        quality_score = max(0, 1 - (critical_issues * 0.2 + (total_issues - critical_issues) * 0.05))
        fact_df['DataQualityScore'] = quality_score
        logger.info(f"Data quality score: {quality_score:.2f}")
    
    logger.info(f"Prepared enhanced fact table with {len(fact_df):,} records")
    
    # ===== STEP 6: Final Data Validation and Quality Assessment =====
    logger.info("Step 6: Final data validation and quality assessment...")
    
    # Final validation checks
    validation_issues = []
    
    # Check for negative revenues
    invalid_revenue = (fact_df['TotalRevenue'] <= 0).sum()
    if invalid_revenue > 0:
        validation_issues.append(f"Found {invalid_revenue} records with invalid revenue")
        fact_df = fact_df[fact_df['TotalRevenue'] > 0]
    
    # Check for missing critical data
    missing_critical = fact_df[['InvoiceNo', 'Quantity', 'UnitPrice']].isnull().any(axis=1).sum()
    if missing_critical > 0:
        validation_issues.append(f"Found {missing_critical} records with missing critical data")
    
    # Check for extreme values
    extreme_quantity = (fact_df['Quantity'] > 1000).sum()
    if extreme_quantity > 0:
        validation_issues.append(f"Found {extreme_quantity} records with extreme quantities")
    
    extreme_price = (fact_df['UnitPrice'] > 1000).sum()
    if extreme_price > 0:
        validation_issues.append(f"Found {extreme_price} records with extreme prices")
    
    # Log validation issues
    if validation_issues:
        logger.warning("Validation issues found:")
        for issue in validation_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All validation checks passed!")
    
    # Generate final data quality report
    logger.info("Generating final data quality report...")
    quality_summary = dq_framework.get_quality_summary()
    
    # Export quality report
    quality_report_path = dq_framework.export_quality_report()
    logger.info(f"Quality report exported to: {quality_report_path}")
    
    # Log quality summary
    logger.info(f"Data Quality Summary:")
    logger.info(f"  - Total checks: {quality_summary['total_checks']}")
    logger.info(f"  - Failed checks: {quality_summary['failed_checks']}")
    logger.info(f"  - Success rate: {quality_summary['success_rate']:.2%}")
    
    if quality_summary['critical_issues']:
        logger.warning(f"  - Critical issues: {len(quality_summary['critical_issues'])}")
    
    logger.info("TRANSFORM phase completed successfully!")
    logger.info(f"Final fact table shape: {fact_df.shape}")
    
    return fact_df, dim_date, dim_customer, dim_product, dim_location, dim_channel, dq_framework.quality_results


def build_enhanced_dim_date(dates):
    """
    Build enhanced Date dimension table with comprehensive time attributes.
    
    Args:
        dates: array-like of datetime objects
    
    Returns:
        pandas.DataFrame with enhanced date dimension attributes
    """
    df = pd.DataFrame({'FullDate': pd.to_datetime(dates).date})
    df = df.drop_duplicates()
    df = df.sort_values('FullDate')
    df.reset_index(drop=True, inplace=True)
    
    # Extract basic date components
    df['Day'] = pd.to_datetime(df['FullDate']).dt.day
    df['Month'] = pd.to_datetime(df['FullDate']).dt.month
    df['Year'] = pd.to_datetime(df['FullDate']).dt.year
    df['Quarter'] = pd.to_datetime(df['FullDate']).dt.quarter
    df['DayOfWeek'] = pd.to_datetime(df['FullDate']).dt.dayofweek  # Monday=0
    df['DayOfYear'] = pd.to_datetime(df['FullDate']).dt.dayofyear
    df['WeekNumber'] = pd.to_datetime(df['FullDate']).dt.isocalendar().week
    
    # String representations
    df['DayName'] = pd.to_datetime(df['FullDate']).dt.day_name().str[:3]  # Mon, Tue, etc.
    df['MonthName'] = pd.to_datetime(df['FullDate']).dt.month_name()
    df['QuarterName'] = 'Q' + df['Quarter'].astype(str)
    
    # Business attributes
    df['IsWeekend'] = pd.to_datetime(df['FullDate']).dt.dayofweek >= 5
    df['IsBusinessDay'] = ~df['IsWeekend']
    df['IsHoliday'] = False  # Could be enhanced with holiday calendar
    
    # Fiscal year (assuming fiscal year starts in April)
    df['FiscalYear'] = df.apply(lambda x: x['Year'] if x['Month'] >= 4 else x['Year'] - 1, axis=1)
    df['FiscalQuarter'] = df.apply(lambda x: ((x['Month'] - 1) // 3) + 1 if x['Month'] >= 4 else ((x['Month'] + 8) // 3) + 1, axis=1)
    df['FiscalMonth'] = df.apply(lambda x: x['Month'] - 3 if x['Month'] >= 4 else x['Month'] + 9, axis=1)
    
    # Seasonality
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Peak season indicator (December for retail)
    df['IsPeakSeason'] = df['Month'] == 12
    
    # Add DateKey (surrogate key)
    df['DateKey'] = df.index + 1
    
    return df


def build_enhanced_dim_location(df):
    """
    Build enhanced Location dimension table with geographic hierarchy.
    
    Args:
        df: pandas.DataFrame with raw data
    
    Returns:
        pandas.DataFrame with enhanced location dimension
    """
    df_location = df[['Country']].drop_duplicates()
    df_location.reset_index(drop=True, inplace=True)
    
    # Add LocationKey (surrogate key)
    df_location['LocationKey'] = df_location.index + 1
    
    # Enhanced geographic hierarchy (simplified mapping)
    country_mapping = {
        'United Kingdom': {'Region': 'Europe', 'Continent': 'Europe', 'Currency': 'GBP', 'IsPrimaryMarket': True},
        'Germany': {'Region': 'Europe', 'Continent': 'Europe', 'Currency': 'EUR', 'IsPrimaryMarket': False},
        'France': {'Region': 'Europe', 'Continent': 'Europe', 'Currency': 'EUR', 'IsPrimaryMarket': False},
        'Netherlands': {'Region': 'Europe', 'Continent': 'Europe', 'Currency': 'EUR', 'IsPrimaryMarket': False},
        'EIRE': {'Region': 'Europe', 'Continent': 'Europe', 'Currency': 'EUR', 'IsPrimaryMarket': False},
        'Australia': {'Region': 'Oceania', 'Continent': 'Oceania', 'Currency': 'AUD', 'IsPrimaryMarket': False},
        'USA': {'Region': 'North America', 'Continent': 'North America', 'Currency': 'USD', 'IsPrimaryMarket': False},
        'Canada': {'Region': 'North America', 'Continent': 'North America', 'Currency': 'CAD', 'IsPrimaryMarket': False},
    }
    
    # Apply mapping
    for country, attributes in country_mapping.items():
        mask = df_location['Country'] == country
        for attr, value in attributes.items():
            df_location.loc[mask, attr] = value
    
    # Fill missing values with defaults
    df_location['Region'] = df_location['Region'].fillna('Unknown')
    df_location['Continent'] = df_location['Continent'].fillna('Unknown')
    df_location['Currency'] = df_location['Currency'].fillna('Unknown')
    df_location['IsPrimaryMarket'] = df_location['IsPrimaryMarket'].fillna(False)
    
    # Add additional attributes
    df_location['City'] = 'Unknown'
    df_location['PostalCode'] = 'Unknown'
    df_location['Latitude'] = None
    df_location['Longitude'] = None
    df_location['GDP'] = None
    df_location['Population'] = None
    df_location['TimeZone'] = 'Unknown'
    df_location['MarketSize'] = 'Unknown'
    df_location['MarketMaturity'] = 'Unknown'
    
    return df_location


def build_enhanced_dim_customer(df):
    """
    Build enhanced Customer dimension table with segmentation and lifecycle.
    
    Args:
        df: pandas.DataFrame with raw data
    
    Returns:
        pandas.DataFrame with enhanced customer dimension
    """
    # Calculate customer metrics first
    customer_metrics = df.groupby('CustomerID').agg({
        'Country': 'first',
        'InvoiceDate': ['min', 'max', 'count'],
        'TotalRevenue': ['sum', 'mean'],
        'Quantity': 'sum'
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['CustomerID', 'Country', 'FirstPurchaseDate', 'LastPurchaseDate', 'TotalOrders', 'TotalRevenue', 'AverageOrderValue', 'TotalQuantity']
    
    # Convert date columns to datetime
    customer_metrics['FirstPurchaseDate'] = pd.to_datetime(customer_metrics['FirstPurchaseDate'])
    customer_metrics['LastPurchaseDate'] = pd.to_datetime(customer_metrics['LastPurchaseDate'])
    
    # Calculate additional metrics
    # Handle NaT values in LastPurchaseDate
    customer_metrics['CustomerRecency'] = customer_metrics['LastPurchaseDate'].apply(
        lambda x: (datetime.now().date() - x.date()).days if pd.notna(x) else None
    )
    customer_metrics['CustomerFrequency'] = customer_metrics['TotalOrders']
    customer_metrics['CustomerMonetary'] = customer_metrics['TotalRevenue']
    
    # Calculate Customer Lifetime Value (simplified)
    customer_metrics['CustomerLifetimeValue'] = customer_metrics['TotalRevenue']
    
    # Create customer segments based on RFM analysis
    # Recency segments
    customer_metrics['RecencySegment'] = pd.cut(
        customer_metrics['CustomerRecency'], 
        bins=[0, 30, 90, 180, float('inf')], 
        labels=['Champions', 'Loyal', 'Potential', 'At Risk']
    )
    
    # Frequency segments
    customer_metrics['FrequencySegment'] = pd.cut(
        customer_metrics['CustomerFrequency'], 
        bins=[0, 1, 3, 10, float('inf')], 
        labels=['New', 'Occasional', 'Regular', 'VIP']
    )
    
    # Monetary segments
    customer_metrics['MonetarySegment'] = pd.cut(
        customer_metrics['CustomerMonetary'], 
        bins=[0, 100, 500, 1000, float('inf')], 
        labels=['Low', 'Medium', 'High', 'Premium']
    )
    
    # Overall customer segment
    customer_metrics['CustomerSegment'] = customer_metrics['RecencySegment'].astype(str) + '_' + customer_metrics['MonetarySegment'].astype(str)
    
    # Customer tier based on CLV
    customer_metrics['CustomerTier'] = pd.cut(
        customer_metrics['CustomerLifetimeValue'], 
        bins=[0, 200, 500, 1000, float('inf')], 
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    
    # Customer status
    customer_metrics['CustomerStatus'] = 'Active'
    customer_metrics.loc[customer_metrics['CustomerRecency'] > 365, 'CustomerStatus'] = 'Inactive'
    customer_metrics.loc[customer_metrics['CustomerRecency'] > 730, 'CustomerStatus'] = 'Churned'
    
    # Add additional attributes
    customer_metrics['IsActive'] = customer_metrics['CustomerStatus'] == 'Active'
    customer_metrics['AgeGroup'] = 'Unknown'
    customer_metrics['Gender'] = 'Unknown'
    
    # Add CustomerKey (surrogate key)
    customer_metrics['CustomerKey'] = customer_metrics.index + 1
    
    # Add Unknown customer with key=0
    unknown_customer = pd.DataFrame({
        'CustomerKey': [0],
        'CustomerID': ['UNKNOWN'],
        'Country': ['UNKNOWN'],
        'FirstPurchaseDate': [pd.NaT],
        'LastPurchaseDate': [pd.NaT],
        'TotalOrders': [0],
        'TotalRevenue': [0],
        'AverageOrderValue': [0],
        'TotalQuantity': [0],
        'CustomerRecency': [None],
        'CustomerFrequency': [0],
        'CustomerMonetary': [0],
        'CustomerLifetimeValue': [0],
        'RecencySegment': ['Unknown'],
        'FrequencySegment': ['Unknown'],
        'MonetarySegment': ['Unknown'],
        'CustomerSegment': ['Unknown'],
        'CustomerTier': ['Unknown'],
        'CustomerStatus': ['Unknown'],
        'IsActive': [False],
        'AgeGroup': ['Unknown'],
        'Gender': ['Unknown']
    })
    
    df_customer = pd.concat([unknown_customer, customer_metrics]).reset_index(drop=True)
    
    return df_customer


def build_enhanced_dim_product(df):
    """
    Build enhanced Product dimension table with categorization and performance metrics.
    
    Args:
        df: pandas.DataFrame with raw data
    
    Returns:
        pandas.DataFrame with enhanced product dimension
    """
    # Calculate product metrics first
    product_metrics = df.groupby('StockCode').agg({
        'Description': 'first',
        'TotalRevenue': ['sum', 'mean'],
        'Quantity': ['sum', 'mean'],
        'UnitPrice': ['mean', 'min', 'max'],
        'InvoiceNo': 'nunique'
    }).reset_index()
    
    # Flatten column names
    product_metrics.columns = ['StockCode', 'Description', 'TotalRevenue', 'AverageRevenue', 'TotalQuantitySold', 'AverageOrderQuantity', 'AveragePrice', 'MinPrice', 'MaxPrice', 'UniqueOrders']
    
    # Clean up description
    product_metrics['Description'] = product_metrics['Description'].fillna('').str.strip()
    
    # Add ProductKey (surrogate key)
    product_metrics['ProductKey'] = product_metrics.index + 1
    
    # Enhanced product categorization based on description keywords
    def categorize_product(description):
        if pd.isna(description) or description == '':
            return 'Unknown', 'Unknown', 'Unknown'
        
        desc_lower = description.lower()
        
        # Category mapping based on keywords
        if any(word in desc_lower for word in ['bag', 'handbag', 'purse', 'wallet']):
            return 'Bags', 'Handbags', 'Accessories'
        elif any(word in desc_lower for word in ['t-shirt', 'shirt', 'top', 'blouse']):
            return 'Clothing', 'Tops', 'Apparel'
        elif any(word in desc_lower for word in ['candle', 'holder', 'light', 'lamp']):
            return 'Home', 'Lighting', 'Decor'
        elif any(word in desc_lower for word in ['box', 'storage', 'container']):
            return 'Home', 'Storage', 'Organization'
        elif any(word in desc_lower for word in ['card', 'paper', 'stationery']):
            return 'Office', 'Stationery', 'Supplies'
        elif any(word in desc_lower for word in ['toy', 'game', 'puzzle']):
            return 'Toys', 'Games', 'Entertainment'
        else:
            return 'General', 'Miscellaneous', 'Other'
    
    # Apply categorization
    categorization = product_metrics['Description'].apply(categorize_product)
    product_metrics['ProductCategory'] = [cat[0] for cat in categorization]
    product_metrics['ProductSubcategory'] = [cat[1] for cat in categorization]
    product_metrics['ProductLine'] = [cat[2] for cat in categorization]
    
    # Add additional attributes
    product_metrics['ProductBrand'] = 'Unknown'
    product_metrics['StandardPrice'] = product_metrics['AveragePrice']
    product_metrics['CostPrice'] = product_metrics['AveragePrice'] * 0.6  # Assume 40% margin
    product_metrics['ProfitMargin'] = ((product_metrics['StandardPrice'] - product_metrics['CostPrice']) / product_metrics['StandardPrice'] * 100).round(2)
    product_metrics['ProductSize'] = 'Unknown'
    product_metrics['ProductColor'] = 'Unknown'
    product_metrics['ProductMaterial'] = 'Unknown'
    product_metrics['ProductWeight'] = None
    product_metrics['IsActive'] = True
    product_metrics['ProductStatus'] = 'Active'
    product_metrics['LaunchDate'] = None
    product_metrics['DiscontinuationDate'] = None
    
    return product_metrics


def build_dim_channel():
    """
    Build Channel dimension table.
    
    Returns:
        pandas.DataFrame with channel dimension
    """
    # For this dataset, we only have online retail data
    df_channel = pd.DataFrame({
        'ChannelKey': [1],
        'ChannelCode': ['ONLINE'],
        'ChannelName': ['Online Retail'],
        'ChannelType': ['Online'],
        'ChannelDescription': ['Online retail channel'],
        'IsActive': [True],
        'LaunchDate': [datetime(2010, 1, 1)],
        'CostPerAcquisition': [10.0],
        'ConversionRate': [2.5]
    })
    
    return df_channel


# ==================== ENHANCED LOAD PHASE ====================

def load_data(fact_df, dim_date, dim_customer, dim_product, dim_location, dim_channel, quality_results=None):
    """
    Enhanced data loading with comprehensive monitoring and error handling.
    Tables must be loaded in specific order to maintain referential integrity.
    
    Args:
        fact_df: Fact table data
        dim_date: Date dimension data
        dim_customer: Customer dimension data
        dim_product: Product dimension data
        dim_location: Location dimension data
        dim_channel: Channel dimension data
        quality_results: Data quality check results
    """
    logger.info("Starting enhanced LOAD phase...")
    
    engine = create_engine(DATABASE_URL, echo=False)
    
    # Create tables if they don't exist
    logger.info("Creating database schema...")
    Base.metadata.create_all(engine)
    
    # Load dimension tables first (in dependency order)
    logger.info("Loading dimension tables...")
    
    try:
        # Load DimDate
        dim_date_columns = ['DateKey', 'FullDate', 'Day', 'Month', 'Year', 'Quarter', 
                           'DayOfWeek', 'DayOfYear', 'WeekNumber', 'DayName', 'MonthName', 
                           'QuarterName', 'IsWeekend', 'IsBusinessDay', 'IsHoliday', 
                           'FiscalYear', 'FiscalQuarter', 'FiscalMonth', 'Season', 'IsPeakSeason']
        
        available_columns = [col for col in dim_date_columns if col in dim_date.columns]
        dim_date[available_columns].to_sql(
            'DimDate', engine, if_exists='replace', index=False, chunksize=BATCH_SIZE
        )
        logger.info(f"Loaded DimDate: {len(dim_date)} records")
        
        # Load DimLocation
        dim_location_columns = ['LocationKey', 'Country', 'Region', 'Continent', 'City', 
                               'PostalCode', 'Latitude', 'Longitude', 'GDP', 'Population', 
                               'Currency', 'TimeZone', 'MarketSize', 'MarketMaturity', 'IsPrimaryMarket']
        
        available_columns = [col for col in dim_location_columns if col in dim_location.columns]
        dim_location[available_columns].to_sql(
            'DimLocation', engine, if_exists='replace', index=False, chunksize=BATCH_SIZE
        )
        logger.info(f"Loaded DimLocation: {len(dim_location)} records")
        
        # Load DimCustomer
        dim_customer_columns = ['CustomerKey', 'CustomerID', 'Country', 'Region', 'Continent', 
                               'City', 'CustomerSegment', 'CustomerTier', 'CustomerLifetimeValue', 
                               'CustomerRecency', 'CustomerFrequency', 'CustomerMonetary', 
                               'FirstPurchaseDate', 'LastPurchaseDate', 'TotalOrders', 'TotalRevenue', 
                               'AverageOrderValue', 'AgeGroup', 'Gender', 'IsActive', 'CustomerStatus']
        
        available_columns = [col for col in dim_customer_columns if col in dim_customer.columns]
        dim_customer[available_columns].to_sql(
            'DimCustomer', engine, if_exists='replace', index=False, chunksize=BATCH_SIZE
        )
        logger.info(f"Loaded DimCustomer: {len(dim_customer)} records")
        
        # Load DimProduct
        dim_product_columns = ['ProductKey', 'StockCode', 'Description', 'ProductCategory', 
                              'ProductSubcategory', 'ProductLine', 'ProductBrand', 'StandardPrice', 
                              'CostPrice', 'ProfitMargin', 'ProductSize', 'ProductColor', 
                              'ProductMaterial', 'ProductWeight', 'TotalQuantitySold', 'TotalRevenue', 
                              'AverageOrderQuantity', 'IsActive', 'ProductStatus', 'LaunchDate', 'DiscontinuationDate']
        
        available_columns = [col for col in dim_product_columns if col in dim_product.columns]
        dim_product[available_columns].to_sql(
            'DimProduct', engine, if_exists='replace', index=False, chunksize=BATCH_SIZE
        )
        logger.info(f"Loaded DimProduct: {len(dim_product)} records")
        
        # Load DimChannel
        dim_channel_columns = ['ChannelKey', 'ChannelCode', 'ChannelName', 'ChannelType', 
                              'ChannelDescription', 'IsActive', 'LaunchDate', 'CostPerAcquisition', 'ConversionRate']
        
        available_columns = [col for col in dim_channel_columns if col in dim_channel.columns]
        dim_channel[available_columns].to_sql(
            'DimChannel', engine, if_exists='replace', index=False, chunksize=BATCH_SIZE
        )
        logger.info(f"Loaded DimChannel: {len(dim_channel)} records")
        
        # Load fact table
        logger.info("Loading fact table...")
        
        fact_columns = ['DateKey', 'CustomerKey', 'ProductKey', 'LocationKey', 'ChannelKey',
                       'InvoiceNo', 'LineItemNo', 'Quantity', 'UnitPrice', 'TotalRevenue',
                       'CostAmount', 'GrossProfit', 'GrossMargin', 'DiscountAmount', 'NetRevenue',
                       'TaxAmount', 'OrderQuantity', 'ReturnQuantity', 'ReturnAmount',
                       'DaysToShip', 'DaysToDeliver', 'LoadTimestamp', 'SourceSystem', 'DataQualityScore']
        
        available_columns = [col for col in fact_columns if col in fact_df.columns]
        fact_df[available_columns].to_sql(
            'FactSales', engine, if_exists='replace', index=False, chunksize=BATCH_SIZE
        )
        logger.info(f"Loaded FactSales: {len(fact_df):,} records")
        
        # Load data quality results if available
        if quality_results:
            logger.info("Loading data quality results...")
            quality_df = pd.DataFrame([
                {
                    'check_name': r.check_name,
                    'table_name': r.table_name,
                    'check_type': r.check_type.value,
                    'severity': r.severity.value,
                    'records_checked': r.records_checked,
                    'records_failed': r.records_failed,
                    'failure_rate': r.failure_rate,
                    'error_message': r.error_message,
                    'check_timestamp': r.check_timestamp,
                    'etl_run_id': r.etl_run_id,
                    'source_system': r.source_system
                }
                for r in quality_results
            ])
            
            quality_df.to_sql(
                'DataQualityLog', engine, if_exists='append', index=False, chunksize=BATCH_SIZE
            )
            logger.info(f"Loaded DataQualityLog: {len(quality_df)} records")
        
        # Create indexes for performance
        logger.info("Creating database indexes...")
        with engine.connect() as conn:
            # Create indexes for fact table
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_date ON FactSales (DateKey)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_customer ON FactSales (CustomerKey)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_product ON FactSales (ProductKey)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_location ON FactSales (LocationKey)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_invoice ON FactSales (InvoiceNo)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_revenue ON FactSales (TotalRevenue)"))
            
            # Create indexes for dimension tables
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_date_year_month ON DimDate (Year, Month)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_customer_segment ON DimCustomer (CustomerSegment)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_product_category ON DimProduct (ProductCategory)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_location_country ON DimLocation (Country)"))
            
            conn.commit()
        
        logger.info("Database indexes created successfully!")
        
        # Generate load summary
        load_summary = {
            'etl_run_id': ETL_RUN_ID,
            'load_timestamp': datetime.now().isoformat(),
            'tables_loaded': {
                'DimDate': int(len(dim_date)),
                'DimLocation': int(len(dim_location)),
                'DimCustomer': int(len(dim_customer)),
                'DimProduct': int(len(dim_product)),
                'DimChannel': int(len(dim_channel)),
                'FactSales': int(len(fact_df))
            },
            'total_records': int(len(dim_date) + len(dim_location) + len(dim_customer) + len(dim_product) + len(dim_channel) + len(fact_df)),
            'data_quality_checks': int(len(quality_results)) if quality_results else 0
        }
        
        # Save load summary
        with open(f'load_summary_{ETL_RUN_ID}.json', 'w') as f:
            json.dump(load_summary, f, indent=2, default=str)
        
        logger.info("LOAD phase completed successfully!")
        logger.info("=" * 80)
        logger.info("ENHANCED ETL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total records loaded: {load_summary['total_records']:,}")
        logger.info(f"ETL Run ID: {ETL_RUN_ID}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during LOAD phase: {e}")
        raise


# ==================== MAIN ETL PIPELINE ====================

def run_etl(file_path=None, chunk_size=None):
    """
    Execute the complete enhanced ETL pipeline.
    
    Args:
        file_path: Optional path to source data file
        chunk_size: Optional chunk size for processing large files
    """
    logger.info("=" * 80)
    logger.info("STARTING ENHANCED ETL PIPELINE")
    logger.info("=" * 80)
    logger.info(f"ETL Run ID: {ETL_RUN_ID}")
    logger.info(f"Start Time: {datetime.now().isoformat()}")
    
    start_time = datetime.now()
    
    try:
        # EXTRACT
        logger.info("Phase 1: EXTRACT")
        df_raw = extract_data(file_path, chunk_size)
        
        # TRANSFORM
        logger.info("Phase 2: TRANSFORM")
        fact_df, dim_date, dim_customer, dim_product, dim_location, dim_channel, quality_results = transform_data(df_raw)
        
        # LOAD
        logger.info("Phase 3: LOAD")
        load_data(fact_df, dim_date, dim_customer, dim_product, dim_location, dim_channel, quality_results)
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("ENHANCED ETL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        logger.info(f"ETL Run ID: {ETL_RUN_ID}")
        logger.info(f"End Time: {end_time.isoformat()}")
        logger.info("=" * 80)
        
        return {
            'status': 'success',
            'etl_run_id': ETL_RUN_ID,
            'execution_time': float(execution_time),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'records_processed': int(len(fact_df)),
            'quality_checks': int(len(quality_results)) if quality_results else 0
        }
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.error("=" * 80)
        logger.error("ETL PIPELINE FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error(f"Execution Time: {execution_time:.2f} seconds")
        logger.error(f"ETL Run ID: {ETL_RUN_ID}")
        logger.error("=" * 80)
        
        return {
            'status': 'failed',
            'etl_run_id': ETL_RUN_ID,
            'execution_time': float(execution_time),
            'error': str(e),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }


if __name__ == '__main__':
    run_etl()

