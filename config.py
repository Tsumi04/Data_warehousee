"""
Configuration file for the Retail Data Warehouse project.
Contains database paths, file paths, and other constants.
"""

import os
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).parent

# Database Configuration
DATABASE_NAME = "retail_dwh_new.db"
DATABASE_PATH = PROJECT_ROOT / DATABASE_NAME

# Data Files
RAW_DATA_FILE = "online_retail.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / RAW_DATA_FILE

# Create necessary directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Connection String
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# ETL Configuration
BATCH_SIZE = 10000  # Process data in batches
CHUNK_SIZE = 10000  # Read CSV in chunks

# Data Quality Thresholds
MIN_QUANTITY = 1
MAX_QUANTITY = 100000
MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 50000.00

# Dashboard Configuration
DASHBOARD_TITLE = "📊 Retail Analytics Dashboard"
DASHBOARD_LAYOUT = "wide"
DEFAULT_START_DATE = "2010-12-01"
DEFAULT_END_DATE = "2011-12-09"

# Colors and Styling
CHART_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'info': '#9467bd'
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "etl.log"

