#!/usr/bin/env python3
"""
Continue System Script
Continues the system startup from where it left off after ETL completion
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continue_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_data_quality():
    """Validate data quality with fresh connections"""
    logger.info("Validating data quality...")
    
    try:
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        
        # Create fresh engine
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
        
        # Check tables
        tables = ['FactSales', 'DimDate', 'DimCustomer', 'DimProduct', 'DimLocation']
        table_counts = {}
        
        for table in tables:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                count = result.fetchone()[0]
                table_counts[table] = count
                logger.info(f"{table}: {count:,} records")
        
        # Quick quality check
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
        
        logger.info("Data quality validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data quality validation error: {e}")
        return False

def run_ml_analytics():
    """Run ML analytics"""
    logger.info("Running ML Analytics...")
    
    try:
        from ml_analytics_engine import MLAnalyticsEngine
        ml_engine = MLAnalyticsEngine()
        results = ml_engine.run_complete_ml_analysis()
        
        logger.info("ML Analytics completed successfully")
        return True
        
    except Exception as e:
        logger.warning(f"ML Analytics failed: {e}")
        logger.info("Continuing without ML Analytics...")
        return True

def start_dashboard():
    """Start the dashboard"""
    logger.info("Starting dashboard...")
    
    try:
        import subprocess
        import sys
        
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
        
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true"
        ])
        
        logger.info("Dashboard started successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Dashboard startup error: {e}")
        return False

def generate_system_report():
    """Generate system report"""
    logger.info("Generating system report...")
    
    try:
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        import json
        from datetime import datetime
        
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'operational',
            'database_stats': {},
            'performance_metrics': {}
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

def main():
    """Main function to continue system startup"""
    print("=" * 60)
    print("CONTINUING RETAIL DATA WAREHOUSE SYSTEM")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Validate data quality
    if not validate_data_quality():
        logger.error("Data quality validation failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Run ML Analytics (optional)
    run_ml_analytics()
    
    # Step 3: Generate system report
    report = generate_system_report()
    
    # Step 4: Start dashboard
    if not start_dashboard():
        logger.error("Dashboard startup failed. Exiting.")
        sys.exit(1)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("=" * 60)
    print("✅ SYSTEM CONTINUED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Execution time: {execution_time:.2f} seconds")
    print("Dashboard: http://localhost:8501")
    print("System Report: system_report.json")
    print("=" * 60)
    
    if report:
        total_records = sum(report['database_stats'].values())
        print(f"Total records in database: {total_records:,}")
    
    print("Press Ctrl+C to stop the system")
    print("=" * 60)
    
    # Keep system running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("System stopped by user")
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
