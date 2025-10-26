"""
Optimized Production Retail Data Warehouse System
Improved version with faster initialization and better error handling
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import json
import concurrent.futures
from typing import Dict, List, Tuple, Any

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

class OptimizedProductionSystem:
    """Optimized production system with parallel processing and caching"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_status = {
            'dependencies': False,
            'data_file': False,
            'database': False,
            'etl': False,
            'data_quality': False,
            'ml_analytics': False,
            'dashboard': False
        }
        self.performance_metrics = {}
    
    def check_dependencies_parallel(self) -> bool:
        """Check dependencies with parallel processing"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'sqlalchemy', 'streamlit', 
            'plotly', 'tqdm', 'sklearn', 'statsmodels'
        ]
        
        def check_package(package):
            try:
                __import__(package)
                return package, True
            except ImportError:
                return package, False
        
        # Check packages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(check_package, required_packages))
        
        missing_packages = [pkg for pkg, status in results if not status]
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Install with: pip install -r requirements.txt")
            return False
        
        logger.info("All dependencies are installed")
        self.system_status['dependencies'] = True
        return True
    
    def check_data_file_optimized(self) -> bool:
        """Optimized data file check with size validation"""
        logger.info("Checking data file...")
        
        data_file = Path("data/raw/online_retail.csv")
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            logger.info("Download dataset from: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
            return False
        
        # Quick file size check
        file_size = data_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Data file: {data_file} ({file_size:.2f} MB)")
        
        # Validate file is not empty and has expected structure
        try:
            import pandas as pd
            # Read only first few rows to validate structure
            sample_df = pd.read_csv(data_file, encoding='ISO-8859-1', nrows=5)
            expected_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
            
            if not all(col in sample_df.columns for col in expected_columns):
                logger.error("Data file structure is invalid")
                return False
                
        except Exception as e:
            logger.error(f"Error validating data file: {e}")
            return False
        
        self.system_status['data_file'] = True
        self.performance_metrics['data_file_size_mb'] = file_size
        return True
    
    def setup_database_optimized(self) -> bool:
        """Optimized database setup with connection pooling"""
        logger.info("Setting up database schema...")
        
        try:
            from database_setup import create_database
            from config import DATABASE_URL
            
            # Check if database already exists and has data
            from sqlalchemy import create_engine, text
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            
            with engine.connect() as conn:
                # Check if tables exist and have data
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='FactSales'"))
                if result.fetchone():
                    # Check if data exists
                    result = conn.execute(text("SELECT COUNT(*) FROM FactSales"))
                    count = result.fetchone()[0]
                    if count > 0:
                        logger.info(f"Database already exists with {count:,} records")
                        self.system_status['database'] = True
                        return True
            
            # Create database if needed
            create_database()
            logger.info("Database schema created")
            self.system_status['database'] = True
            return True
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            return False
    
    def run_etl_pipeline_optimized(self) -> bool:
        """Optimized ETL pipeline with progress tracking"""
        logger.info("Starting optimized ETL pipeline...")
        
        try:
            # Check if ETL has already been run recently
            etl_log_file = Path("etl.log")
            if etl_log_file.exists():
                # Check last ETL run time
                stat = etl_log_file.stat()
                last_modified = datetime.fromtimestamp(stat.st_mtime)
                time_diff = datetime.now() - last_modified
                
                if time_diff.total_seconds() < 3600:  # Less than 1 hour ago
                    logger.info("ETL was run recently, skipping...")
                    self.system_status['etl'] = True
                    return True
            
            # Run ETL with progress tracking
            from etl import run_etl
            result = run_etl()
            
            if result['status'] == 'success':
                logger.info("ETL pipeline completed successfully")
                logger.info(f"Records processed: {result['records_processed']:,}")
                logger.info(f"Execution time: {result['execution_time']:.2f} seconds")
                
                self.system_status['etl'] = True
                self.performance_metrics['etl_records_processed'] = result['records_processed']
                self.performance_metrics['etl_execution_time'] = result['execution_time']
                return True
            else:
                logger.error(f"ETL pipeline failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"ETL pipeline error: {e}")
            return False
    
    def validate_data_quality_optimized(self) -> bool:
        """Optimized data quality validation with sampling"""
        logger.info("Validating data quality...")
        
        try:
            from sqlalchemy import create_engine, text
            from config import DATABASE_URL
            
            # Create a fresh engine connection
            engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
            
            # Check tables with optimized queries
            tables = ['FactSales', 'DimDate', 'DimCustomer', 'DimProduct', 'DimLocation']
            table_counts = {}
            
            # Use separate connection for each operation to avoid connection issues
            for table in tables:
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                        count = result.fetchone()[0]
                        table_counts[table] = count
                        logger.info(f"{table}: {count:,} records")
                        
                        if count == 0:
                            logger.error(f"Table {table} is empty!")
                            return False
                except Exception as e:
                    logger.error(f"Error checking table {table}: {e}")
                    return False
            
            # Quick data quality check with fresh connection
            try:
                with engine.connect() as conn:
                    quality_query = """
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT CustomerKey) as unique_customers,
                        COUNT(DISTINCT ProductKey) as unique_products,
                        SUM(TotalRevenue) as total_revenue
                    FROM FactSales
                    """
                    
                    result = conn.execute(text(quality_query))
                    quality_stats = result.fetchone()
                    
                    logger.info(f"Data quality check: {quality_stats[0]:,} total records, "
                               f"{quality_stats[1]:,} customers, {quality_stats[2]:,} products, "
                               f"${quality_stats[3]:,.2f} revenue")
            except Exception as e:
                logger.warning(f"Could not run quality check query: {e}")
                # Don't fail the entire validation for this
            
            self.system_status['data_quality'] = True
            self.performance_metrics['table_counts'] = table_counts
            logger.info("Data quality validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data quality validation error: {e}")
            return False
    
    def run_ml_analytics_optimized(self) -> bool:
        """Optimized ML analytics with error handling"""
        logger.info("Running ML Analytics...")
        
        try:
            # Check if ML models already exist
            ml_models_dir = Path("ml_models")
            if ml_models_dir.exists() and any(ml_models_dir.iterdir()):
                logger.info("ML models already exist, skipping training...")
                self.system_status['ml_analytics'] = True
                return True
            
            from ml_analytics_engine import MLAnalyticsEngine
            ml_engine = MLAnalyticsEngine()
            
            # Run ML analysis with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ml_engine.run_complete_ml_analysis)
                try:
                    results = future.result(timeout=300)  # 5 minute timeout
                    logger.info("ML Analytics completed successfully")
                    self.system_status['ml_analytics'] = True
                    return True
                except concurrent.futures.TimeoutError:
                    logger.warning("ML Analytics timed out, continuing without it...")
                    return True
                    
        except Exception as e:
            logger.warning(f"ML Analytics failed: {e}")
            logger.info("Continuing without ML Analytics...")
            return True  # Don't fail the entire system for ML issues
    
    def start_dashboard_optimized(self) -> bool:
        """Optimized dashboard startup with health check"""
        logger.info("Starting dashboard...")
        
        try:
            # Check dashboard files
            dashboard_files = ["advanced_dashboard.py", "app.py"]
            dashboard_file = None
            
            for file in dashboard_files:
                if Path(file).exists():
                    dashboard_file = file
                    break
            
            if not dashboard_file:
                logger.error("No dashboard file found")
                return False
            
            logger.info(f"Starting Streamlit dashboard: {dashboard_file}")
            logger.info("Dashboard will be available at: http://localhost:8501")
            
            # Start Streamlit in background
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", dashboard_file,
                "--server.port", "8501",
                "--server.address", "localhost",
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give dashboard time to start
            time.sleep(3)
            
            # Check if dashboard is running
            try:
                import requests
                response = requests.get("http://localhost:8501", timeout=5)
                if response.status_code == 200:
                    logger.info("Dashboard started successfully")
                    self.system_status['dashboard'] = True
                    return True
            except:
                logger.warning("Could not verify dashboard status, but process started")
                self.system_status['dashboard'] = True
                return True
            
        except Exception as e:
            logger.error(f"Dashboard startup error: {e}")
            return False
    
    def generate_system_report_optimized(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        logger.info("Generating system report...")
        
        try:
            from sqlalchemy import create_engine, text
            from config import DATABASE_URL
            
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            
            end_time = time.time()
            total_execution_time = end_time - self.start_time
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status,
                'performance_metrics': self.performance_metrics,
                'execution_time_seconds': total_execution_time,
                'database_stats': {},
                'system_health': 'operational' if all(self.system_status.values()) else 'degraded'
            }
            
            # Get database statistics
            tables = ['FactSales', 'DimDate', 'DimCustomer', 'DimProduct', 'DimLocation']
            
            with engine.connect() as conn:
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                    count = result.fetchone()[0]
                    report['database_stats'][table] = count
            
            # Save report
            with open('system_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info("System report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def run_system(self) -> bool:
        """Run the complete optimized system"""
        logger.info("=" * 80)
        logger.info("OPTIMIZED PRODUCTION RETAIL DATA WAREHOUSE SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Step 1: Check dependencies
        if not self.check_dependencies_parallel():
            logger.error("Dependency check failed. Exiting.")
            return False
        
        # Step 2: Check data file
        if not self.check_data_file_optimized():
            logger.error("Data file check failed. Exiting.")
            return False
        
        # Step 3: Setup database
        if not self.setup_database_optimized():
            logger.error("Database setup failed. Exiting.")
            return False
        
        # Step 4: Run ETL pipeline
        if not self.run_etl_pipeline_optimized():
            logger.error("ETL pipeline failed. Exiting.")
            return False
        
        # Step 5: Validate data quality
        if not self.validate_data_quality_optimized():
            logger.error("Data quality validation failed. Exiting.")
            return False
        
        # Step 6: Run ML Analytics (optional)
        self.run_ml_analytics_optimized()
        
        # Step 7: Generate system report
        report = self.generate_system_report_optimized()
        
        # Step 8: Start dashboard
        if not self.start_dashboard_optimized():
            logger.error("Dashboard startup failed. Exiting.")
            return False
        
        # Final status
        end_time = time.time()
        total_execution_time = end_time - self.start_time
        
        logger.info("=" * 80)
        logger.info("SYSTEM STARTED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
        logger.info("Dashboard: http://localhost:8501")
        logger.info("System Report: system_report.json")
        logger.info("Logs: production_system.log")
        logger.info("=" * 80)
        
        if report:
            total_records = sum(report['database_stats'].values())
            logger.info(f"Total records in database: {total_records:,}")
        
        logger.info("Press Ctrl+C to stop the system")
        logger.info("=" * 80)
        
        return True

def main():
    """Main function to run the optimized system"""
    system = OptimizedProductionSystem()
    
    try:
        success = system.run_system()
        if success:
            # Keep system running
            while True:
                time.sleep(60)  # Check every minute
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("System stopped by user")
        logger.info("Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
