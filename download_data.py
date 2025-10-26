"""
Script to download the Online Retail dataset from Kaggle
"""

import kaggle
import os
import zipfile
from pathlib import Path

from config import RAW_DATA_DIR, RAW_DATA_FILE

def download_dataset():
    """
    Download the Online Retail dataset from Kaggle.
    Requires kaggle.json to be placed in ~/.kaggle/ directory.
    """
    
    print("=" * 80)
    print("DOWNLOADING ONLINE RETAIL DATASET FROM KAGGLE")
    print("=" * 80)
    
    # Ensure download directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kaggle dataset path
    dataset = "thedevastator/online-retail-sales-and-customer-data"
    
    try:
        print(f"Downloading dataset: {dataset}")
        print(f"Target directory: {RAW_DATA_DIR}")
        
        # Download dataset
        kaggle.api.dataset_download_files(dataset, path=str(RAW_DATA_DIR), unzip=True)
        
        print("\n✅ Dataset downloaded successfully!")
        print(f"Please ensure the CSV file is named: {RAW_DATA_FILE}")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nPlease ensure:")
        print("1. You have kaggle.json in your ~/.kaggle/ directory")
        print("2. You have accepted the dataset rules on Kaggle")
        print("3. Your Kaggle API credentials are valid")
        
        print("\nAlternatively, manually download from:")
        print("https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
        print(f"And place the CSV in: {RAW_DATA_DIR}")

if __name__ == '__main__':
    download_dataset()


