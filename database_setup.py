"""
Database Setup Module for Retail Data Warehouse
Creates the Star Schema with Dimension and Fact tables using SQLAlchemy ORM.
"""

import sys
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Boolean, Numeric, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
import logging

from config import DATABASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Base class for declarative models
Base = declarative_base()


# ==================== DIMENSION TABLES ====================

class DimDate(Base):
    """Date Dimension Table - Time dimension for date-based analysis"""
    __tablename__ = 'DimDate'
    
    DateKey = Column(Integer, primary_key=True)
    FullDate = Column(Date, nullable=False)
    Day = Column(Integer, nullable=False)
    Month = Column(Integer, nullable=False)
    Year = Column(Integer, nullable=False)
    Quarter = Column(Integer, nullable=False)
    DayOfWeek = Column(Integer, nullable=False)
    DayName = Column(String(10))
    MonthName = Column(String(10))
    IsWeekend = Column(Boolean, default=False)
    IsHoliday = Column(Boolean, default=False)
    WeekNumber = Column(Integer)
    
    def __repr__(self):
        return f"<DimDate(DateKey={self.DateKey}, FullDate={self.FullDate})>"


class DimCustomer(Base):
    """Customer Dimension Table - Stores customer attributes"""
    __tablename__ = 'DimCustomer'
    
    CustomerKey = Column(Integer, primary_key=True)
    CustomerID = Column(String(255))
    Country = Column(String(255))
    CustomerSegment = Column(String(50))  # High/Medium/Low value
    
    def __repr__(self):
        return f"<DimCustomer(CustomerKey={self.CustomerKey}, CustomerID={self.CustomerID})>"


class DimProduct(Base):
    """Product Dimension Table - Stores product attributes"""
    __tablename__ = 'DimProduct'
    
    ProductKey = Column(Integer, primary_key=True)
    StockCode = Column(String(255))
    Description = Column(Text)
    ProductCategory = Column(String(100))
    ProductSubcategory = Column(String(100))
    
    def __repr__(self):
        return f"<DimProduct(ProductKey={self.ProductKey}, StockCode={self.StockCode})>"


class DimLocation(Base):
    """Location Dimension Table - Geographic dimension"""
    __tablename__ = 'DimLocation'
    
    LocationKey = Column(Integer, primary_key=True)
    Country = Column(String(255), unique=True)
    Region = Column(String(100))
    Continent = Column(String(50))
    
    def __repr__(self):
        return f"<DimLocation(LocationKey={self.LocationKey}, Country={self.Country})>"


# ==================== FACT TABLE ====================

class FactSales(Base):
    """Sales Fact Table - Central fact table containing business measures"""
    __tablename__ = 'FactSales'
    
    SalesKey = Column(Integer, primary_key=True, autoincrement=True)
    DateKey = Column(Integer, ForeignKey('DimDate.DateKey'), nullable=False)
    CustomerKey = Column(Integer, ForeignKey('DimCustomer.CustomerKey'), nullable=False)
    ProductKey = Column(Integer, ForeignKey('DimProduct.ProductKey'), nullable=False)
    LocationKey = Column(Integer, ForeignKey('DimLocation.LocationKey'), nullable=False)
    
    # Natural Keys from source system
    InvoiceNo = Column(String(255))
    
    # Business Measures
    Quantity = Column(Integer, nullable=False)
    UnitPrice = Column(Numeric(10, 2), nullable=False)
    TotalRevenue = Column(Numeric(12, 2), nullable=False)
    
    # Additional measures
    DiscountAmount = Column(Numeric(10, 2), default=0)
    NetRevenue = Column(Numeric(12, 2))
    
    # ETL metadata
    LoadTimestamp = Column(DateTime)
    
    def __repr__(self):
        return f"<FactSales(SalesKey={self.SalesKey}, TotalRevenue={self.TotalRevenue})>"


class DataQualityLog(Base):
    """Data Quality Log table for tracking quality check results"""
    __tablename__ = 'DataQualityLog'
    
    LogKey = Column(Integer, primary_key=True, autoincrement=True)
    check_name = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=False)
    check_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    records_checked = Column(Integer, nullable=False)
    records_failed = Column(Integer, nullable=False)
    failure_rate = Column(Numeric(5, 4), nullable=False)
    error_message = Column(Text)
    check_timestamp = Column(DateTime, nullable=False)
    etl_run_id = Column(String(50), nullable=False)
    source_system = Column(String(100), nullable=False)
    
    def __repr__(self):
        return f"<DataQualityLog(check_name={self.check_name}, severity={self.severity})>"


# ==================== HELPER FUNCTIONS ====================

def create_database(engine=None):
    """
    Create the data warehouse database and all tables.
    
    Args:
        engine: Optional SQLAlchemy engine. If None, creates a new engine.
    """
    if engine is None:
        engine = create_engine(DATABASE_URL, echo=False)
    
    logger.info(f"Creating database at: {DATABASE_URL}")
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    logger.info("Database and all tables created successfully!")
    logger.info(f"Created tables: {', '.join(Base.metadata.tables.keys())}")
    
    return engine


def get_session(engine=None):
    """
    Get a database session.
    
    Args:
        engine: Optional SQLAlchemy engine. If None, creates a new engine.
    
    Returns:
        SQLAlchemy session object
    """
    if engine is None:
        engine = create_engine(DATABASE_URL, echo=False)
    
    Session = sessionmaker(bind=engine)
    return Session()


def print_schema():
    """Print the database schema for documentation."""
    print("\n" + "="*80)
    print("DATABASE SCHEMA - STAR SCHEMA")
    print("="*80)
    
    for table_name, table in Base.metadata.tables.items():
        print(f"\n{table_name}")
        print("-" * 80)
        for column in table.columns:
            print(f"  {column.name:25s} {str(column.type):20s} {('PK' if column.primary_key else '')}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # Create database
    logger.info("Starting database setup...")
    
    try:
        create_database()
        print_schema()
        
        logger.info("="*80)
        logger.info("DATABASE SETUP COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Database Location: {DATABASE_URL}")
        
    except Exception as e:
        logger.error(f"Error during database setup: {e}")
        sys.exit(1)

