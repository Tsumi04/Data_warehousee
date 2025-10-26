#!/usr/bin/env python3
"""
Optimized Startup Script for Retail Data Warehouse
Quick startup with performance optimizations and error handling
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
        logging.FileHandler('startup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def quick_system_check():
    """Quick system check before starting"""
    logger.info("Performing quick system check...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        return False
    
    # Check if we're in the right directory
    required_files = ['config.py', 'database_setup.py', 'etl.py']
    for file in required_files:
        if not Path(file).exists():
            logger.error(f"Required file not found: {file}")
            return False
    
    # Check data file
    data_file = Path("data/raw/online_retail.csv")
    if not data_file.exists():
        logger.warning("Data file not found. System will create sample data if needed.")
    
    logger.info("System check passed")
    return True

def start_optimized_system():
    """Start the optimized production system"""
    logger.info("Starting optimized production system...")
    
    try:
        from optimized_production_system import OptimizedProductionSystem
        
        system = OptimizedProductionSystem()
        success = system.run_system()
        
        if success:
            logger.info("System started successfully!")
            return True
        else:
            logger.error("System startup failed")
            return False
            
    except ImportError:
        logger.error("Optimized production system not found. Falling back to standard system...")
        return start_standard_system()
    except Exception as e:
        logger.error(f"Error starting optimized system: {e}")
        return False

def start_standard_system():
    """Fallback to standard production system"""
    logger.info("Starting standard production system...")
    
    try:
        from run_production_system import main
        main()
        return True
    except Exception as e:
        logger.error(f"Error starting standard system: {e}")
        return False

def start_dashboard_only():
    """Start only the dashboard (for development)"""
    logger.info("Starting dashboard only...")
    
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
        
        logger.info(f"Starting dashboard: {dashboard_file}")
        logger.info("Dashboard will be available at: http://localhost:8501")
        
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        logger.info("Dashboard started successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 RETAIL DATA WAREHOUSE - OPTIMIZED STARTUP")
    print("=" * 60)
    
    # Quick system check
    if not quick_system_check():
        logger.error("System check failed. Exiting.")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "dashboard":
            success = start_dashboard_only()
        elif mode == "standard":
            success = start_standard_system()
        elif mode == "optimized":
            success = start_optimized_system()
        else:
            print("Usage: python start_optimized_system.py [dashboard|standard|optimized]")
            print("  dashboard  - Start only the dashboard")
            print("  standard   - Start standard production system")
            print("  optimized  - Start optimized production system (default)")
            sys.exit(1)
    else:
        # Default to optimized system
        success = start_optimized_system()
    
    if not success:
        logger.error("Startup failed!")
        sys.exit(1)
    
    print("=" * 60)
    print("✅ SYSTEM STARTED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
