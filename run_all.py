"""
Master script to run the complete pipeline
"""

import os
import sys
import logging

from database_setup import create_database, print_schema
from etl import run_etl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Run the complete pipeline: Database Setup -> ETL -> Ready for Dashboard
    """
    logger.info("=" * 80)
    logger.info("RETAIL DATA WAREHOUSE - COMPLETE PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Step 1: Setup Database
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATABASE SETUP")
        logger.info("="*80)
        create_database()
        print_schema()
        
        # Step 2: Run ETL
        logger.info("\n" + "="*80)
        logger.info("STEP 2: ETL PIPELINE")
        logger.info("="*80)
        run_etl()
        
        # Step 3: Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("Next steps:")
        logger.info("1. Run: streamlit run app.py")
        logger.info("2. Open browser to: http://localhost:8501")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()



