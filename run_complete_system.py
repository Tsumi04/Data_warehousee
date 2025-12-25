#!/usr/bin/env python3
"""
Complete Retail Data Warehouse System Runner
This script orchestrates the entire data warehouse system including:
- Database setup
- ETL pipeline execution
- Advanced analytics generation
- Dashboard launch
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetailDataWarehouseSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.logger = logger
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        self.logger.info("Checking prerequisites...")
        
        # Check if data file exists
        data_file = Path("data/raw/online_retail.csv")
        if not data_file.exists():
            self.logger.error("Source data file not found!")
            self.logger.info("Please download the dataset from: https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data")
            return False
        
        # Check if required Python packages are installed
        required_packages = [
            'pandas', 'numpy', 'sqlalchemy', 'streamlit', 
            'plotly', 'great-expectations', 'python-dotenv'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing packages: {missing_packages}")
            self.logger.info("Please install missing packages: pip install " + " ".join(missing_packages))
            return False
        
        self.logger.info("All prerequisites met!")
        return True
    
    def setup_database(self):
        """Setup database schema"""
        self.logger.info("Setting up database...")
        
        try:
            from database_setup import create_database
            engine = create_database()
            self.logger.info("Database setup completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            return False
    
    def run_etl_pipeline(self):
        """Run ETL pipeline"""
        self.logger.info("Starting ETL pipeline...")
        
        try:
            from etl import run_etl
            result = run_etl()
            
            if result['status'] == 'success':
                self.logger.info(f"ETL pipeline completed successfully!")
                self.logger.info(f"Records processed: {result['records_processed']:,}")
                self.logger.info(f"Execution time: {result['execution_time']:.2f} seconds")
                return True
            else:
                self.logger.error(f"ETL pipeline failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}")
            return False
    
    def generate_advanced_analytics(self):
        """Generate advanced analytics"""
        self.logger.info("Generating advanced analytics...")
        
        try:
            from advanced_analytics import AdvancedAnalytics
            analytics = AdvancedAnalytics()
            
            # Generate insights report
            insights = analytics.generate_insights_report()
            
            if insights:
                self.logger.info("Advanced analytics generated successfully!")
                self.logger.info(f"Insights generated for {len(insights)} categories")
                return True
            else:
                self.logger.warning("No insights generated")
                return False
                
        except Exception as e:
            self.logger.error(f"Advanced analytics generation failed: {e}")
            return False
    
    def launch_dashboard(self):
        """Launch the dashboard"""
        self.logger.info("Launching dashboard...")
        
        try:
            # Check if streamlit is available
            import streamlit
            
            # Launch dashboard
            self.logger.info("Starting Streamlit dashboard...")
            self.logger.info("Dashboard will be available at: http://localhost:8501")
            
            # Run streamlit in subprocess
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "advanced_dashboard.py", "--server.port", "8501"
            ])
            
        except Exception as e:
            self.logger.error(f"Dashboard launch failed: {e}")
            return False
    
    def run_system_tests(self):
        """Run system tests"""
        self.logger.info("Running system tests...")
        
        try:
            # Import and run tests
            import unittest
            from test_system import TestDataWarehouseSystem
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(TestDataWarehouseSystem)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            if result.wasSuccessful():
                self.logger.info("All system tests passed!")
                return True
            else:
                self.logger.warning(f"Some tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
                return False
                
        except Exception as e:
            self.logger.error(f"System tests failed: {e}")
            return False
    
    def generate_system_report(self):
        """Generate system report"""
        self.logger.info("Generating system report...")
        
        end_time = datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()
        
        report = {
            'system_start_time': self.start_time.isoformat(),
            'system_end_time': end_time.isoformat(),
            'total_execution_time': execution_time,
            'status': 'completed'
        }
        
        # Save report
        import json
        with open('system_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"System report saved to system_report.json")
        self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
    
    def run_complete_system(self, skip_tests=False, launch_dashboard=True):
        """Run the complete system"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING RETAIL DATA WAREHOUSE SYSTEM")
        self.logger.info("=" * 60)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed. Exiting.")
            return False
        
        # Step 2: Setup database
        if not self.setup_database():
            self.logger.error("Database setup failed. Exiting.")
            return False
        
        # Step 3: Run ETL pipeline
        if not self.run_etl_pipeline():
            self.logger.error("ETL pipeline failed. Exiting.")
            return False
        
        # Step 4: Generate advanced analytics
        if not self.generate_advanced_analytics():
            self.logger.warning("Advanced analytics generation failed, but continuing...")
        
        # Step 5: Run system tests (optional)
        if not skip_tests:
            if not self.run_system_tests():
                self.logger.warning("Some system tests failed, but continuing...")
        
        # Step 6: Generate system report
        self.generate_system_report()
        
        # Step 7: Launch dashboard (optional)
        if launch_dashboard:
            self.logger.info("System setup completed successfully!")
            self.logger.info("Launching dashboard...")
            self.launch_dashboard()
        else:
            self.logger.info("System setup completed successfully!")
            self.logger.info("To launch dashboard, run: streamlit run advanced_dashboard.py")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Retail Data Warehouse System')
    parser.add_argument('--skip-tests', action='store_true', help='Skip system tests')
    parser.add_argument('--no-dashboard', action='store_true', help='Do not launch dashboard')
    parser.add_argument('--etl-only', action='store_true', help='Run ETL pipeline only')
    parser.add_argument('--dashboard-only', action='store_true', help='Launch dashboard only')
    
    args = parser.parse_args()
    
    system = RetailDataWarehouseSystem()
    
    if args.etl_only:
        # Run ETL pipeline only
        if system.check_prerequisites() and system.setup_database():
            system.run_etl_pipeline()
    elif args.dashboard_only:
        # Launch dashboard only
        system.launch_dashboard()
    else:
        # Run complete system
        system.run_complete_system(
            skip_tests=args.skip_tests,
            launch_dashboard=not args.no_dashboard
        )

if __name__ == "__main__":
    main()


