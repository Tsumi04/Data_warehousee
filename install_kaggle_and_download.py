"""
Automatic script to install Kaggle API and download the real dataset
"""

import subprocess
import sys
import os
import zipfile
from pathlib import Path

def install_kaggle():
    """Install Kaggle API"""
    print("Installing Kaggle API...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✅ Kaggle API installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error installing Kaggle: {e}")
        return False


def download_dataset():
    """Download the real dataset"""
    print("\n" + "="*80)
    print("DOWNLOADING REAL ONLINE RETAIL DATASET")
    print("="*80)
    
    dataset = "thedevastator/online-retail-sales-and-customer-data"
    
    try:
        # Change to raw directory
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        os.chdir(str(raw_dir))
        
        # Download using kaggle command
        subprocess.check_call([
            sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset
        ])
        
        print("✅ Download complete!")
        
        # Find and extract zip file
        zip_files = list(Path(".").glob("*.zip"))
        if zip_files:
            zip_file = zip_files[0]
            print(f"Extracting: {zip_file.name}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            zip_file.unlink()  # Delete zip
            
            # Find CSV file
            csv_files = list(Path(".").glob("*.csv"))
            if csv_files:
                main_csv = csv_files[0]
                # Rename to online_retail.csv if needed
                if main_csv.name != "online_retail.csv":
                    target = Path("online_retail.csv")
                    if target.exists():
                        target.unlink()
                    main_csv.rename(target)
                print(f"✅ Dataset ready: {target}")
            
        os.chdir("../..")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        os.chdir("../..")
        return False


def main():
    """Main function"""
    print("="*80)
    print("SETUP REAL DATASET FROM KAGGLE")
    print("="*80)
    
    # Check kaggle credentials
    kaggle_dir = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_dir.exists():
        print("\n⚠️  WARNING: Kaggle credentials not found!")
        print("\nPlease:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Click 'Create New Token'")
        print("3. Download kaggle.json")
        print("4. Place it in: ~/.kaggle/kaggle.json")
        print("\nOr use manual download (see HUONG_DAN_DOWNLOAD_REAL_DATA.md)")
        return
    
    # Install kaggle if needed
    try:
        import kaggle
    except ImportError:
        if not install_kaggle():
            return
    
    # Download dataset
    if download_dataset():
        print("\n" + "="*80)
        print("✅ REAL DATASET DOWNLOADED!")
        print("="*80)
        print("\nNext steps:")
        print("1. Delete old sample data: del data\\raw\\online_retail.csv")
        print("2. Run ETL: python etl.py")
        print("3. Start dashboard: python -m streamlit run app.py")
        print("="*80)
    else:
        print("\n❌ Download failed!")
        print("Please see: HUONG_DAN_DOWNLOAD_REAL_DATA.md")


if __name__ == '__main__':
    main()


