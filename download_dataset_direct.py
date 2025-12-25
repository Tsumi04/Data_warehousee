"""
Download Online Retail dataset directly without Kaggle API
This script will try to download from a public source or guide the user.
"""

import os
import requests
from pathlib import Path
import zipfile

def download_dataset():
    """
    Download the Online Retail dataset.
    Note: Kaggle datasets require login. This script will guide the user.
    """
    print("="*80)
    print("DOWNLOADING ONLINE RETAIL DATASET")
    print("="*80)
    
    # Check if user has downloaded manually
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    existing_csv = list(raw_dir.glob("*.csv"))
    
    if existing_csv and len([f for f in existing_csv if f.stat().st_size > 10000]) > 0:
        print("\nFound CSV file(s) in data/raw:")
        for f in existing_csv:
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
        print("\nUsing existing file(s)!")
        return True
    
    print("\nKaggle requires login to download datasets.")
    print("Please follow these steps:\n")
    print("1. Go to: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
    print("2. Click the 'Download' button")
    print("3. Login to Kaggle (create account if needed)")
    print("4. Extract the ZIP file")
    print("5. Copy the CSV file to: data/raw/online_retail.csv")
    print("\nAfter copying the file, run: python etl.py")
    print("="*80)
    
    return False


if __name__ == '__main__':
    download_dataset()



