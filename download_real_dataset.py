"""
Download the real Online Retail dataset from Kaggle
"""

import os
import zipfile
from pathlib import Path
import subprocess
import sys

def download_with_kaggle_api():
    """Download dataset using Kaggle API"""
    print("="*80)
    print("DOWNLOADING REAL DATASET FROM KAGGLE")
    print("="*80)
    
    dataset = "thedevastator/online-retail-sales-and-customer-data"
    
    try:
        print(f"Downloading: {dataset}")
        print("Target: data/raw/")
        
        # Check if kaggle.json exists
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            print("\nERROR: Kaggle API credentials not found!")
            print("Please:")
            print("1. Go to: https://www.kaggle.com/account")
            print("2. Create API token (kaggle.json)")
            print("3. Place it in: ~/.kaggle/kaggle.json")
            print("\nOR use manual download method (see below)")
            return False
        
        # Run kaggle command
        result = subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset
        ], cwd="data/raw", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Download successful!")
            
            # Extract zip file
            zip_files = list(Path("data/raw").glob("*.zip"))
            if zip_files:
                print(f"Extracting: {zip_files[0].name}")
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    zip_ref.extractall("data/raw")
                zip_files[0].unlink()  # Delete zip file
                print("Extraction complete!")
            
            print("\nDataset ready in: data/raw/")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("\nERROR: Kaggle CLI not installed!")
        print("Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def show_manual_instructions():
    """Show manual download instructions"""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("\n1. Go to: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
    print("2. Click 'Download' button")
    print("3. Extract the ZIP file")
    print("4. Find the CSV file (online_retail.csv or similar)")
    print("5. Copy it to: data/raw/online_retail.csv")
    print("\nThen run: python etl.py")
    print("="*80)


if __name__ == '__main__':
    success = download_with_kaggle_api()
    
    if not success:
        show_manual_instructions()
    else:
        print("\n" + "="*80)
        print("SUCCESS! Real dataset downloaded!")
        print("Next: python etl.py")
        print("="*80)


