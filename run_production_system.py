"""
Production Retail Data Warehouse System
Lệnh chạy chính cho toàn bộ hệ thống Data Warehouse production-ready
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import json

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Kiểm tra các thư viện cần thiết"""
    print("Kiem tra dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sqlalchemy', 'streamlit', 
        'plotly', 'tqdm', 'sklearn', 'statsmodels'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Thieu packages: {missing_packages}")
        print("Cai dat: pip install -r requirements.txt")
        return False
    
    print("Tat ca dependencies da duoc cai dat")
    return True

def check_data_file():
    """Kiểm tra file dữ liệu nguồn"""
    print("Kiem tra file du lieu...")
    
    data_file = Path("data/raw/online_retail.csv")
    
    if not data_file.exists():
        print(f"Khong tim thay file: {data_file}")
        print("Tai dataset tu: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
        return False
    
    file_size = data_file.stat().st_size / (1024 * 1024)  # MB
    print(f"File du lieu: {data_file} ({file_size:.2f} MB)")
    return True

def setup_database():
    """Thiết lập database schema"""
    print("Thiet lap database schema...")
    
    try:
        from database_setup import create_database
        create_database()
        print("Database schema da duoc tao")
        return True
    except Exception as e:
        print(f"Loi thiet lap database: {e}")
        return False

def run_etl_pipeline():
    """Chạy ETL pipeline"""
    print("Bat dau ETL pipeline...")
    
    try:
        # Sử dụng basic ETL pipeline
        from etl import run_etl
        result = run_etl()
        
        if result['status'] == 'success':
            print("ETL pipeline hoan thanh")
            print(f"Records processed: {result['records_processed']:,}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            return True
        else:
            print(f"ETL pipeline that bai: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Loi ETL pipeline: {e}")
        return False

def validate_data_quality():
    """Kiểm tra chất lượng dữ liệu"""
    print("Kiem tra chat luong du lieu...")
    
    try:
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        
        engine = create_engine(DATABASE_URL)
        
        # Kiểm tra các bảng
        tables = ['FactSales', 'DimDate', 'DimCustomer', 'DimProduct', 'DimLocation']
        
        for table in tables:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                count = result.fetchone()[0]
                print(f"{table}: {count:,} records")
                
                if count == 0:
                    print(f"Bang {table} trong!")
                    return False
        
        print("Kiem tra chat luong du lieu hoan thanh")
        return True
        
    except Exception as e:
        print(f"Loi kiem tra chat luong: {e}")
        return False

def run_ml_analytics():
    """Chạy ML analytics (optional)"""
    print("Chay ML Analytics...")
    
    try:
        from ml_analytics_engine import MLAnalyticsEngine
        ml_engine = MLAnalyticsEngine()
        results = ml_engine.run_complete_ml_analysis()
        
        print("ML Analytics hoan thanh")
        return True
        
    except Exception as e:
        print(f"ML Analytics khong chay duoc: {e}")
        return False

def start_dashboard():
    """Khởi động dashboard"""
    print("Khoi dong dashboard...")
    
    try:
        # Kiểm tra file dashboard
        dashboard_files = ["advanced_dashboard.py", "app.py"]
        dashboard_file = None
        
        for file in dashboard_files:
            if Path(file).exists():
                dashboard_file = file
                break
        
        if not dashboard_file:
            print("Khong tim thay file dashboard")
            return False
        
        print(f"Khoi dong Streamlit dashboard: {dashboard_file}")
        print("Dashboard se co tai: http://localhost:8501")
        
        # Khởi động Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        print("Dashboard da khoi dong")
        return True
        
    except Exception as e:
        print(f"Loi khoi dong dashboard: {e}")
        return False

def generate_system_report():
    """Tạo báo cáo hệ thống"""
    print("Tao bao cao he thong...")
    
    try:
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        
        engine = create_engine(DATABASE_URL)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'operational',
            'database_stats': {},
            'performance_metrics': {}
        }
        
        # Thống kê database
        tables = ['FactSales', 'DimDate', 'DimCustomer', 'DimProduct', 'DimLocation']
        
        with engine.connect() as conn:
            for table in tables:
                result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                count = result.fetchone()[0]
                report['database_stats'][table] = count
        
        # Lưu báo cáo
        with open('system_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("Bao cao he thong da duoc tao")
        return report
        
    except Exception as e:
        print(f"Loi tao bao cao: {e}")
        return None

def main():
    """Hàm chính để chạy toàn bộ hệ thống"""
    print("=" * 80)
    print("PRODUCTION RETAIL DATA WAREHOUSE SYSTEM")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Bước 1: Kiểm tra dependencies
    if not check_dependencies():
        print("Kiem tra dependencies that bai. Thoat.")
        sys.exit(1)
    
    # Bước 2: Kiểm tra file dữ liệu
    if not check_data_file():
        print("Kiem tra file du lieu that bai. Thoat.")
        sys.exit(1)
    
    # Bước 3: Thiết lập database
    if not setup_database():
        print("Thiet lap database that bai. Thoat.")
        sys.exit(1)
    
    # Bước 4: Chạy ETL pipeline
    if not run_etl_pipeline():
        print("ETL pipeline that bai. Thoat.")
        sys.exit(1)
    
    # Bước 5: Kiểm tra chất lượng dữ liệu
    if not validate_data_quality():
        print("Kiem tra chat luong du lieu that bai. Thoat.")
        sys.exit(1)
    
    # Bước 6: Chạy ML Analytics (optional)
    run_ml_analytics()
    
    # Bước 7: Tạo báo cáo hệ thống
    report = generate_system_report()
    if report:
        total_records = sum(report['database_stats'].values())
        print(f"Tong records: {total_records:,}")
    
    # Bước 8: Khởi động dashboard
    if not start_dashboard():
        print("Khoi dong dashboard that bai. Thoat.")
        sys.exit(1)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("=" * 80)
    print("HE THONG DA KHOI DONG THANH CONG!")
    print("=" * 80)
    print(f"Total execution time: {execution_time:.2f} seconds")
    print("Dashboard: http://localhost:8501")
    print("System Report: system_report.json")
    print("Logs: production_system.log")
    print("=" * 80)
    print("Nhan Ctrl+C de dung he thong")
    print("=" * 80)
    
    # Giữ hệ thống chạy
    try:
        while True:
            time.sleep(60)  # Kiểm tra mỗi phút
    except KeyboardInterrupt:
        print("He thong da duoc dung boi nguoi dung")
        print("\nHe thong da dung. Tam biet!")

if __name__ == '__main__':
    main()