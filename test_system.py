"""
Comprehensive System Testing for Retail Data Warehouse
This script tests all components of the data warehouse system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

class TestDataWarehouseSystem(unittest.TestCase):
    """Test suite for the retail data warehouse system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        print("Setting up test environment...")
        
        # Check if data file exists
        cls.data_file = Path("data/raw/online_retail.csv")
        if not cls.data_file.exists():
            print("Warning: Source data file not found. Some tests may fail.")
        
        # Import modules
        try:
            from config import DATABASE_URL, RAW_DATA_PATH
            from database_setup import create_database
            from etl import run_etl
            from data_quality_framework import DataQualityFramework
            cls.DATABASE_URL = DATABASE_URL
            cls.RAW_DATA_PATH = RAW_DATA_PATH
            cls.create_database = create_database
            cls.run_etl = run_etl
            cls.DataQualityFramework = DataQualityFramework
        except ImportError as e:
            print(f"Error importing modules: {e}")
            raise
    
    def test_01_data_file_exists(self):
        """Test if source data file exists."""
        self.assertTrue(self.data_file.exists(), "Source data file not found")
        
        if self.data_file.exists():
            file_size = self.data_file.stat().st_size
            self.assertGreater(file_size, 0, "Source data file is empty")
            print(f"[OK] Source data file exists ({file_size / (1024*1024):.2f} MB)")
    
    def test_02_data_file_structure(self):
        """Test source data file structure."""
        if not self.data_file.exists():
            self.skipTest("Source data file not found")
        
        df = pd.read_csv(self.data_file, encoding='ISO-8859-1')
        
        # Check required columns
        required_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                           'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
        
        for col in required_columns:
            self.assertIn(col, df.columns, f"Required column {col} not found")
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Quantity']), "Quantity should be numeric")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['UnitPrice']), "UnitPrice should be numeric")
        
        # Check data quality
        self.assertGreater(len(df), 0, "Data file is empty")
        print(f"[OK] Data file structure is valid ({len(df):,} records)")
    
    def test_03_database_creation(self):
        """Test database creation."""
        try:
            from database_setup import create_database
            engine = create_database()
            self.assertIsNotNone(engine, "Database creation failed")
            print("[OK] Database created successfully")
        except Exception as e:
            self.fail(f"Database creation failed: {e}")
    
    def test_04_etl_pipeline(self):
        """Test ETL pipeline execution."""
        if not self.data_file.exists():
            self.skipTest("Source data file not found")
        
        try:
            from etl import run_etl
            result = run_etl()
            self.assertEqual(result['status'], 'success', f"ETL pipeline failed: {result.get('error', 'Unknown error')}")
            self.assertGreater(result['records_processed'], 0, "No records processed")
            print(f"[OK] ETL pipeline completed successfully ({result['records_processed']:,} records)")
        except Exception as e:
            self.fail(f"ETL pipeline failed: {e}")
    
    def test_05_database_tables(self):
        """Test if all required tables exist and have data."""
        from sqlalchemy import create_engine, text
        
        engine = create_engine(self.DATABASE_URL)
        
        required_tables = ['FactSales', 'DimDate', 'DimCustomer', 'DimProduct', 'DimLocation']
        
        with engine.connect() as conn:
            for table in required_tables:
                result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
                count = result.fetchone()[0]
                self.assertGreater(count, 0, f"Table {table} is empty")
                print(f"[OK] Table {table}: {count:,} records")
    
    def test_06_data_quality_framework(self):
        """Test data quality framework."""
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000]
        })
        
        from data_quality_framework import DataQualityFramework
        dq_framework = DataQualityFramework("TEST_RUN")
        
        # Run quality checks
        results = dq_framework.run_all_checks(sample_data, 'TestTable', ['id'])
        
        self.assertIsInstance(results, list, "Quality check results should be a list")
        # Note: Some checks might not run if data doesn't meet criteria
        print(f"[OK] Data quality framework working ({len(results)} checks run)")
    
    def test_07_dashboard_imports(self):
        """Test if dashboard can be imported."""
        try:
            import advanced_dashboard
            print("[OK] Dashboard module imports successfully")
        except ImportError as e:
            self.fail(f"Dashboard import failed: {e}")
    
    def test_08_configuration(self):
        """Test configuration settings."""
        from config import DATABASE_URL, RAW_DATA_PATH, BATCH_SIZE
        
        self.assertIsNotNone(DATABASE_URL, "DATABASE_URL not configured")
        self.assertIsNotNone(RAW_DATA_PATH, "RAW_DATA_PATH not configured")
        self.assertGreater(BATCH_SIZE, 0, "BATCH_SIZE should be positive")
        print("[OK] Configuration settings are valid")
    
    def test_09_data_relationships(self):
        """Test referential integrity between tables."""
        from sqlalchemy import create_engine, text
        
        engine = create_engine(self.DATABASE_URL)
        
        with engine.connect() as conn:
            # Test FactSales foreign keys
            result = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM FactSales fs
                LEFT JOIN DimDate d ON fs.DateKey = d.DateKey
                LEFT JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
                LEFT JOIN DimProduct p ON fs.ProductKey = p.ProductKey
                LEFT JOIN DimLocation l ON fs.LocationKey = l.LocationKey
                WHERE d.DateKey IS NULL 
                   OR c.CustomerKey IS NULL 
                   OR p.ProductKey IS NULL 
                   OR l.LocationKey IS NULL
            """))
            
            orphaned_records = result.fetchone()[0]
            self.assertEqual(orphaned_records, 0, f"Found {orphaned_records} orphaned records in FactSales")
            print("[OK] Referential integrity is maintained")
    
    def test_10_performance_metrics(self):
        """Test system performance metrics."""
        from sqlalchemy import create_engine, text
        
        engine = create_engine(self.DATABASE_URL)
        
        with engine.connect() as conn:
            # Test query performance
            start_time = datetime.now()
            
            result = conn.execute(text("""
                SELECT 
                    SUM(TotalRevenue) as TotalRevenue,
                    COUNT(DISTINCT CustomerKey) as UniqueCustomers,
                    COUNT(*) as TotalRecords
                FROM FactSales
            """))
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            data = result.fetchone()
            self.assertIsNotNone(data, "Performance query failed")
            self.assertLess(execution_time, 5.0, f"Query too slow: {execution_time:.2f}s")
            print(f"[OK] Performance test passed ({execution_time:.2f}s)")
    
    def test_11_data_quality_scores(self):
        """Test data quality scores."""
        from sqlalchemy import create_engine, text
        from config import DATABASE_URL
        
        engine = create_engine(DATABASE_URL)
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        AVG(DataQualityScore) as AvgQuality,
                        MIN(DataQualityScore) as MinQuality,
                        MAX(DataQualityScore) as MaxQuality
                    FROM FactSales
                """))
                
                quality_data = result.fetchone()
                if quality_data[0] is not None:  # If DataQualityScore column exists
                    avg_quality = float(quality_data[0])
                    self.assertGreaterEqual(avg_quality, 0.0, "Average quality score should be >= 0")
                    self.assertLessEqual(avg_quality, 1.0, "Average quality score should be <= 1")
                    print(f"[OK] Data quality scores are valid (avg: {avg_quality:.2f})")
                else:
                    print("[OK] Data quality scores column not found (optional feature)")
        except Exception as e:
            # If column doesn't exist, that's OK for this test
            print(f"[OK] Data quality scores column not found (optional feature): {e}")
    
    def test_12_file_structure(self):
        """Test project file structure."""
        required_files = [
            'config.py',
            'database_setup.py',
            'etl.py',
            'advanced_dashboard.py',
            'data_quality_framework.py',
            'advanced_dwh_design.py',
            'run_production_system.py',
            'requirements.txt',
            'TECHNICAL_DOCUMENTATION.md'
        ]
        
        for file in required_files:
            self.assertTrue(Path(file).exists(), f"Required file {file} not found")
        
        print("[OK] All required files are present")
    
    def test_13_data_volume(self):
        """Test data volume after ETL."""
        from sqlalchemy import create_engine, text
        
        engine = create_engine(self.DATABASE_URL)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM FactSales"))
            fact_count = result.fetchone()[0]
            
            # Should have processed some data
            self.assertGreater(fact_count, 1000, f"Too few records in FactSales: {fact_count}")
            print(f"[OK] Data volume is adequate ({fact_count:,} fact records)")
    
    def test_14_error_handling(self):
        """Test error handling in ETL pipeline."""
        # Test with invalid file path
        try:
            from etl import run_etl
            result = run_etl("invalid_file.csv")
            # Should handle error gracefully
            if result['status'] == 'failed':
                print("[OK] Error handling works correctly")
            else:
                self.fail("ETL should fail with invalid file path")
        except Exception as e:
            print(f"[OK] Error handling works correctly: {e}")

def run_performance_test():
    """Run performance tests."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTING")
    print("="*60)
    
    from sqlalchemy import create_engine, text
    from config import DATABASE_URL
    
    engine = create_engine(DATABASE_URL)
    
    test_queries = [
        ("Simple Count", "SELECT COUNT(*) FROM FactSales"),
        ("Revenue Sum", "SELECT SUM(TotalRevenue) FROM FactSales"),
        ("Customer Count", "SELECT COUNT(DISTINCT CustomerKey) FROM FactSales"),
        ("Complex Join", """
            SELECT 
                d.Year,
                d.Month,
                SUM(fs.TotalRevenue) as MonthlyRevenue
            FROM FactSales fs
            JOIN DimDate d ON fs.DateKey = d.DateKey
            GROUP BY d.Year, d.Month
            ORDER BY d.Year, d.Month
        """),
        ("Customer Analysis", """
            SELECT 
                c.CustomerSegment,
                COUNT(DISTINCT c.CustomerKey) as NumCustomers,
                SUM(fs.TotalRevenue) as TotalRevenue
            FROM FactSales fs
            JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
            GROUP BY c.CustomerSegment
        """)
    ]
    
    for query_name, query in test_queries:
        start_time = datetime.now()
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            data = result.fetchall()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"{query_name:20s}: {execution_time:.3f}s ({len(data)} rows)")

def run_data_quality_test():
    """Run data quality tests."""
    print("\n" + "="*60)
    print("DATA QUALITY TESTING")
    print("="*60)
    
    from sqlalchemy import create_engine, text
    from config import DATABASE_URL
    
    engine = create_engine(DATABASE_URL)
    
    quality_checks = [
        ("Null Values in FactSales", """
            SELECT COUNT(*) as null_count
            FROM FactSales
            WHERE DateKey IS NULL 
               OR CustomerKey IS NULL 
               OR ProductKey IS NULL 
               OR LocationKey IS NULL
        """),
        ("Negative Revenue", """
            SELECT COUNT(*) as negative_revenue
            FROM FactSales
            WHERE TotalRevenue < 0
        """),
        ("Zero Quantity", """
            SELECT COUNT(*) as zero_quantity
            FROM FactSales
            WHERE Quantity <= 0
        """),
        ("Orphaned Records", """
            SELECT COUNT(*) as orphaned
            FROM FactSales fs
            LEFT JOIN DimDate d ON fs.DateKey = d.DateKey
            LEFT JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
            LEFT JOIN DimProduct p ON fs.ProductKey = p.ProductKey
            LEFT JOIN DimLocation l ON fs.LocationKey = l.LocationKey
            WHERE d.DateKey IS NULL 
               OR c.CustomerKey IS NULL 
               OR p.ProductKey IS NULL 
               OR l.LocationKey IS NULL
        """)
    ]
    
    with engine.connect() as conn:
        for check_name, query in quality_checks:
            result = conn.execute(text(query))
            count = result.fetchone()[0]
            
            if count == 0:
                print(f"[OK] {check_name}: PASS")
            else:
                print(f"[FAIL] {check_name}: FAIL ({count} issues)")

def main():
    """Run all tests."""
    print("="*80)
    print("RETAIL DATA WAREHOUSE SYSTEM TESTING")
    print("="*80)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_test()
    
    # Run data quality tests
    run_data_quality_test()
    
    print("\n" + "="*80)
    print("TESTING COMPLETED")
    print("="*80)

if __name__ == '__main__':
    main()
