"""
Advanced Data Warehouse Design for Retail Analytics
This module defines an enhanced Star Schema with additional dimensions and fact tables
for comprehensive retail business intelligence.
"""

from sqlalchemy import Column, Integer, String, Date, DateTime, Boolean, Numeric, Text, ForeignKey, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

# ==================== ENHANCED DIMENSION TABLES ====================

class DimDate(Base):
    """Enhanced Date Dimension with comprehensive time attributes"""
    __tablename__ = 'DimDate'
    
    DateKey = Column(Integer, primary_key=True)
    FullDate = Column(Date, nullable=False, unique=True)
    
    # Basic date components
    Day = Column(Integer, nullable=False)
    Month = Column(Integer, nullable=False)
    Year = Column(Integer, nullable=False)
    Quarter = Column(Integer, nullable=False)
    DayOfWeek = Column(Integer, nullable=False)  # Monday=0, Sunday=6
    DayOfYear = Column(Integer, nullable=False)
    WeekNumber = Column(Integer, nullable=False)
    
    # String representations
    DayName = Column(String(10), nullable=False)
    MonthName = Column(String(10), nullable=False)
    QuarterName = Column(String(10), nullable=False)
    
    # Business attributes
    IsWeekend = Column(Boolean, default=False)
    IsHoliday = Column(Boolean, default=False)
    IsBusinessDay = Column(Boolean, default=True)
    FiscalYear = Column(Integer, nullable=False)
    FiscalQuarter = Column(Integer, nullable=False)
    FiscalMonth = Column(Integer, nullable=False)
    
    # Seasonality
    Season = Column(String(10))  # Spring, Summer, Fall, Winter
    IsPeakSeason = Column(Boolean, default=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_date_year_month', 'Year', 'Month'),
        Index('idx_date_quarter', 'Year', 'Quarter'),
        Index('idx_date_weekend', 'IsWeekend'),
    )


class DimCustomer(Base):
    """Enhanced Customer Dimension with segmentation and lifecycle"""
    __tablename__ = 'DimCustomer'
    
    CustomerKey = Column(Integer, primary_key=True)
    CustomerID = Column(String(255), unique=True)
    
    # Geographic attributes
    Country = Column(String(255), nullable=False)
    Region = Column(String(100))
    Continent = Column(String(50))
    City = Column(String(100))
    
    # Customer segmentation
    CustomerSegment = Column(String(50))  # High/Medium/Low Value
    CustomerTier = Column(String(20))     # Platinum/Gold/Silver/Bronze
    CustomerLifetimeValue = Column(Numeric(12, 2), default=0)
    CustomerRecency = Column(Integer)     # Days since last purchase
    CustomerFrequency = Column(Integer)   # Number of purchases
    CustomerMonetary = Column(Numeric(12, 2))  # Total monetary value
    
    # Behavioral attributes
    FirstPurchaseDate = Column(Date)
    LastPurchaseDate = Column(Date)
    TotalOrders = Column(Integer, default=0)
    TotalRevenue = Column(Numeric(12, 2), default=0)
    AverageOrderValue = Column(Numeric(10, 2), default=0)
    
    # Demographics (if available)
    AgeGroup = Column(String(20))
    Gender = Column(String(10))
    
    # Status
    IsActive = Column(Boolean, default=True)
    CustomerStatus = Column(String(20), default='Active')  # Active/Inactive/Churned
    
    # Indexes
    __table_args__ = (
        Index('idx_customer_segment', 'CustomerSegment'),
        Index('idx_customer_country', 'Country'),
        Index('idx_customer_tier', 'CustomerTier'),
    )


class DimProduct(Base):
    """Enhanced Product Dimension with categorization and pricing"""
    __tablename__ = 'DimProduct'
    
    ProductKey = Column(Integer, primary_key=True)
    StockCode = Column(String(255), unique=True, nullable=False)
    Description = Column(Text)
    
    # Product categorization
    ProductCategory = Column(String(100))
    ProductSubcategory = Column(String(100))
    ProductBrand = Column(String(100))
    ProductLine = Column(String(100))
    
    # Pricing information
    StandardPrice = Column(Numeric(10, 2))
    CostPrice = Column(Numeric(10, 2))
    ProfitMargin = Column(Numeric(5, 2))  # Percentage
    
    # Product attributes
    ProductSize = Column(String(50))
    ProductColor = Column(String(50))
    ProductMaterial = Column(String(100))
    ProductWeight = Column(Numeric(8, 2))
    
    # Performance metrics
    TotalQuantitySold = Column(Integer, default=0)
    TotalRevenue = Column(Numeric(12, 2), default=0)
    AverageOrderQuantity = Column(Numeric(8, 2), default=0)
    
    # Status
    IsActive = Column(Boolean, default=True)
    ProductStatus = Column(String(20), default='Active')
    LaunchDate = Column(Date)
    DiscontinuationDate = Column(Date)
    
    # Indexes
    __table_args__ = (
        Index('idx_product_category', 'ProductCategory'),
        Index('idx_product_brand', 'ProductBrand'),
        Index('idx_product_status', 'IsActive'),
    )


class DimLocation(Base):
    """Enhanced Location Dimension with geographic hierarchy"""
    __tablename__ = 'DimLocation'
    
    LocationKey = Column(Integer, primary_key=True)
    Country = Column(String(255), nullable=False)
    
    # Geographic hierarchy
    Region = Column(String(100))
    Continent = Column(String(50))
    City = Column(String(100))
    PostalCode = Column(String(20))
    
    # Geographic coordinates
    Latitude = Column(Numeric(10, 6))
    Longitude = Column(Numeric(10, 6))
    
    # Economic indicators
    GDP = Column(Numeric(15, 2))
    Population = Column(Integer)
    Currency = Column(String(10))
    TimeZone = Column(String(50))
    
    # Market characteristics
    MarketSize = Column(String(20))  # Large/Medium/Small
    MarketMaturity = Column(String(20))  # Developed/Emerging/Developing
    IsPrimaryMarket = Column(Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_location_country', 'Country'),
        Index('idx_location_region', 'Region'),
        Index('idx_location_continent', 'Continent'),
    )


class DimChannel(Base):
    """Channel Dimension for different sales channels"""
    __tablename__ = 'DimChannel'
    
    ChannelKey = Column(Integer, primary_key=True)
    ChannelCode = Column(String(50), unique=True, nullable=False)
    ChannelName = Column(String(100), nullable=False)
    ChannelType = Column(String(50))  # Online/Offline/Mobile/Phone
    ChannelDescription = Column(Text)
    
    # Channel attributes
    IsActive = Column(Boolean, default=True)
    LaunchDate = Column(Date)
    CostPerAcquisition = Column(Numeric(10, 2))
    ConversionRate = Column(Numeric(5, 2))


# ==================== ENHANCED FACT TABLES ====================

class FactSales(Base):
    """Enhanced Sales Fact Table with comprehensive measures"""
    __tablename__ = 'FactSales'
    
    SalesKey = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    DateKey = Column(Integer, ForeignKey('DimDate.DateKey'), nullable=False)
    CustomerKey = Column(Integer, ForeignKey('DimCustomer.CustomerKey'), nullable=False)
    ProductKey = Column(Integer, ForeignKey('DimProduct.ProductKey'), nullable=False)
    LocationKey = Column(Integer, ForeignKey('DimLocation.LocationKey'), nullable=False)
    ChannelKey = Column(Integer, ForeignKey('DimChannel.ChannelKey'), nullable=False)
    
    # Natural Keys
    InvoiceNo = Column(String(255), nullable=False)
    LineItemNo = Column(Integer, default=1)
    
    # Core Business Measures
    Quantity = Column(Integer, nullable=False)
    UnitPrice = Column(Numeric(10, 2), nullable=False)
    TotalRevenue = Column(Numeric(12, 2), nullable=False)
    
    # Enhanced Financial Measures
    CostAmount = Column(Numeric(10, 2), default=0)
    GrossProfit = Column(Numeric(12, 2), default=0)
    GrossMargin = Column(Numeric(5, 2), default=0)  # Percentage
    DiscountAmount = Column(Numeric(10, 2), default=0)
    NetRevenue = Column(Numeric(12, 2), default=0)
    TaxAmount = Column(Numeric(10, 2), default=0)
    
    # Operational Measures
    OrderQuantity = Column(Integer, default=1)
    ReturnQuantity = Column(Integer, default=0)
    ReturnAmount = Column(Numeric(10, 2), default=0)
    
    # Time-based measures
    DaysToShip = Column(Integer)
    DaysToDeliver = Column(Integer)
    
    # ETL Metadata
    LoadTimestamp = Column(DateTime, default=datetime.utcnow)
    SourceSystem = Column(String(50), default='Online_Retail')
    DataQualityScore = Column(Numeric(3, 2), default=1.0)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_fact_date', 'DateKey'),
        Index('idx_fact_customer', 'CustomerKey'),
        Index('idx_fact_product', 'ProductKey'),
        Index('idx_fact_location', 'LocationKey'),
        Index('idx_fact_invoice', 'InvoiceNo'),
        Index('idx_fact_revenue', 'TotalRevenue'),
    )


class FactInventory(Base):
    """Inventory Fact Table for stock management"""
    __tablename__ = 'FactInventory'
    
    InventoryKey = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    DateKey = Column(Integer, ForeignKey('DimDate.DateKey'), nullable=False)
    ProductKey = Column(Integer, ForeignKey('DimProduct.ProductKey'), nullable=False)
    LocationKey = Column(Integer, ForeignKey('DimLocation.LocationKey'), nullable=False)
    
    # Inventory Measures
    BeginningStock = Column(Integer, default=0)
    StockReceived = Column(Integer, default=0)
    StockSold = Column(Integer, default=0)
    StockReturned = Column(Integer, default=0)
    StockAdjusted = Column(Integer, default=0)
    EndingStock = Column(Integer, default=0)
    
    # Value Measures
    BeginningValue = Column(Numeric(12, 2), default=0)
    EndingValue = Column(Numeric(12, 2), default=0)
    AverageCost = Column(Numeric(10, 2), default=0)
    
    # Performance Metrics
    StockTurnover = Column(Numeric(8, 2), default=0)
    DaysInStock = Column(Integer, default=0)
    StockoutDays = Column(Integer, default=0)
    
    # ETL Metadata
    LoadTimestamp = Column(DateTime, default=datetime.utcnow)
    SourceSystem = Column(String(50), default='Inventory_System')


class FactCustomerBehavior(Base):
    """Customer Behavior Fact Table for advanced analytics"""
    __tablename__ = 'FactCustomerBehavior'
    
    BehaviorKey = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Keys
    DateKey = Column(Integer, ForeignKey('DimDate.DateKey'), nullable=False)
    CustomerKey = Column(Integer, ForeignKey('DimCustomer.CustomerKey'), nullable=False)
    
    # Behavioral Measures
    SessionDuration = Column(Integer)  # Minutes
    PageViews = Column(Integer, default=0)
    Clicks = Column(Integer, default=0)
    BounceRate = Column(Numeric(5, 2), default=0)
    ConversionRate = Column(Numeric(5, 2), default=0)
    
    # Engagement Metrics
    TimeOnSite = Column(Integer)  # Minutes
    ReturnVisits = Column(Integer, default=0)
    CartAbandonmentRate = Column(Numeric(5, 2), default=0)
    
    # ETL Metadata
    LoadTimestamp = Column(DateTime, default=datetime.utcnow)
    SourceSystem = Column(String(50), default='Web_Analytics')


# ==================== DATA QUALITY TABLES ====================

class DataQualityLog(Base):
    """Data Quality Logging Table"""
    __tablename__ = 'DataQualityLog'
    
    LogKey = Column(Integer, primary_key=True, autoincrement=True)
    
    # Quality Check Details
    CheckName = Column(String(100), nullable=False)
    TableName = Column(String(100), nullable=False)
    CheckType = Column(String(50), nullable=False)  # Completeness/Accuracy/Consistency/Validity
    Severity = Column(String(20), nullable=False)   # Critical/High/Medium/Low
    
    # Results
    RecordsChecked = Column(Integer, nullable=False)
    RecordsFailed = Column(Integer, nullable=False)
    FailureRate = Column(Numeric(5, 2), nullable=False)
    ErrorMessage = Column(Text)
    
    # Metadata
    CheckTimestamp = Column(DateTime, default=datetime.utcnow)
    ETLRunID = Column(String(50))
    SourceSystem = Column(String(50))


class DataLineage(Base):
    """Data Lineage Tracking Table"""
    __tablename__ = 'DataLineage'
    
    LineageKey = Column(Integer, primary_key=True, autoincrement=True)
    
    # Source Information
    SourceTable = Column(String(100), nullable=False)
    SourceColumn = Column(String(100), nullable=False)
    SourceRecordCount = Column(Integer)
    
    # Target Information
    TargetTable = Column(String(100), nullable=False)
    TargetColumn = Column(String(100), nullable=False)
    TargetRecordCount = Column(Integer)
    
    # Transformation Details
    TransformationType = Column(String(50))  # Direct/Copy/Calculated/Lookup
    TransformationLogic = Column(Text)
    BusinessRule = Column(Text)
    
    # Metadata
    ETLRunID = Column(String(50), nullable=False)
    ProcessStartTime = Column(DateTime, nullable=False)
    ProcessEndTime = Column(DateTime)
    ProcessDuration = Column(Integer)  # Seconds
    Status = Column(String(20), default='Running')  # Running/Success/Failed


# ==================== AGGREGATED TABLES ====================

class AggDailySales(Base):
    """Daily Sales Aggregation Table for performance"""
    __tablename__ = 'AggDailySales'
    
    AggKey = Column(Integer, primary_key=True, autoincrement=True)
    
    # Dimensions
    DateKey = Column(Integer, ForeignKey('DimDate.DateKey'), nullable=False)
    CustomerKey = Column(Integer, ForeignKey('DimCustomer.CustomerKey'), nullable=False)
    LocationKey = Column(Integer, ForeignKey('DimLocation.LocationKey'), nullable=False)
    
    # Aggregated Measures
    TotalRevenue = Column(Numeric(12, 2), nullable=False)
    TotalQuantity = Column(Integer, nullable=False)
    OrderCount = Column(Integer, nullable=False)
    UniqueProducts = Column(Integer, nullable=False)
    AverageOrderValue = Column(Numeric(10, 2), nullable=False)
    
    # ETL Metadata
    LoadTimestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_agg_daily_date', 'DateKey'),
        Index('idx_agg_daily_customer', 'CustomerKey'),
        Index('idx_agg_daily_location', 'LocationKey'),
    )


class AggMonthlySales(Base):
    """Monthly Sales Aggregation Table"""
    __tablename__ = 'AggMonthlySales'
    
    AggKey = Column(Integer, primary_key=True, autoincrement=True)
    
    # Dimensions
    Year = Column(Integer, nullable=False)
    Month = Column(Integer, nullable=False)
    CustomerKey = Column(Integer, ForeignKey('DimCustomer.CustomerKey'), nullable=False)
    LocationKey = Column(Integer, ForeignKey('DimLocation.LocationKey'), nullable=False)
    
    # Aggregated Measures
    TotalRevenue = Column(Numeric(12, 2), nullable=False)
    TotalQuantity = Column(Integer, nullable=False)
    OrderCount = Column(Integer, nullable=False)
    UniqueCustomers = Column(Integer, nullable=False)
    UniqueProducts = Column(Integer, nullable=False)
    
    # Growth Metrics
    RevenueGrowth = Column(Numeric(5, 2), default=0)  # Month-over-month
    QuantityGrowth = Column(Numeric(5, 2), default=0)
    CustomerGrowth = Column(Numeric(5, 2), default=0)
    
    # ETL Metadata
    LoadTimestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_agg_monthly_year_month', 'Year', 'Month'),
        Index('idx_agg_monthly_customer', 'CustomerKey'),
        Index('idx_agg_monthly_location', 'LocationKey'),
    )


# ==================== HELPER FUNCTIONS ====================

def get_schema_summary():
    """Get a summary of all tables in the schema"""
    tables = {
        'Dimension Tables': [
            'DimDate', 'DimCustomer', 'DimProduct', 'DimLocation', 'DimChannel'
        ],
        'Fact Tables': [
            'FactSales', 'FactInventory', 'FactCustomerBehavior'
        ],
        'Data Quality Tables': [
            'DataQualityLog', 'DataLineage'
        ],
        'Aggregated Tables': [
            'AggDailySales', 'AggMonthlySales'
        ]
    }
    return tables


def print_enhanced_schema():
    """Print the enhanced database schema"""
    print("\n" + "="*100)
    print("ENHANCED DATA WAREHOUSE SCHEMA - PRODUCTION READY")
    print("="*100)
    
    for category, tables in get_schema_summary().items():
        print(f"\n{category}:")
        print("-" * 50)
        for table in tables:
            print(f"  • {table}")
    
    print("\n" + "="*100)
    print("TOTAL TABLES: 12")
    print("DIMENSION TABLES: 5")
    print("FACT TABLES: 3")
    print("DATA QUALITY TABLES: 2")
    print("AGGREGATED TABLES: 2")
    print("="*100)


if __name__ == '__main__':
    print_enhanced_schema()


