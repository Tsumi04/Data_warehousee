"""
Script to check dataset and prepare for ETL
"""

import os
from pathlib import Path

def main():
    print("="*80)
    print("DATASET CHECK AND PREPARATION")
    print("="*80)
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for CSV files
    csv_files = list(raw_dir.glob("*.csv"))
    
    if not csv_files:
        print("\n>>> No CSV file found in data/raw/")
        print("\nYou need to download the dataset from Kaggle:")
        print("1. Visit: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
        print("2. Click 'Download' button")
        print("3. Extract ZIP file")
        print("4. Copy CSV file to: data/raw/online_retail.csv")
        return
    
    # Check file sizes
    print("\nFound CSV file(s):")
    real_files = []
    for f in csv_files:
        size = f.stat().st_size
        print(f"  - {f.name}: {size:,} bytes")
        
        # Check if it's likely a real dataset (>100KB) or sample (<100KB)
        if size > 100000:
            real_files.append(f)
            print(f"    >>> Likely a REAL dataset")
        else:
            print(f"    >>> WARNING: Likely a SAMPLE file")
    
    # Determine what to do
    if len(real_files) > 0:
        print("\n>>> REAL DATASET FOUND!")
        print(f"Using: {real_files[0].name}")
        
        # Rename if needed
        if real_files[0].name != "online_retail.csv":
            target = raw_dir / "online_retail.csv"
            print(f"\nRenaming to: online_retail.csv")
            if target.exists() and target != real_files[0]:
                target.unlink()
            real_files[0].rename(target)
            print(">>> Renamed successfully!")
        
        print("\n>>> Ready to run ETL!")
        print("Run: python etl.py")
        print("(This will take 10-15 minutes with real dataset)")
        
    else:
        print("\n>>> WARNING: Only SAMPLE data found.")
        print("\nTo use REAL dataset:")
        print("1. Download from: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
        print("2. Extract and copy CSV to: data/raw/online_retail.csv")
        print("3. Then run: python etl.py")


if __name__ == '__main__':
    main()

