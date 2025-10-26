#!/usr/bin/env python3
"""
Advanced Analytics Module for Retail Data Warehouse
Provides sophisticated business intelligence and analytics capabilities.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import logging

from config import DATABASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced analytics engine for retail data warehouse"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url)
        self.logger = logging.getLogger(__name__)
    
    def get_rfm_analysis(self) -> pd.DataFrame:
        """
        Perform RFM (Recency, Frequency, Monetary) Analysis
        
        Returns:
            DataFrame with RFM scores and segments
        """
        query = """
        WITH customer_metrics AS (
            SELECT 
                c.CustomerKey,
                c.CustomerID,
                c.Country,
                c.CustomerSegment,
                MAX(d.FullDate) as LastPurchaseDate,
                COUNT(DISTINCT fs.InvoiceNo) as Frequency,
                SUM(fs.TotalRevenue) as Monetary,
                AVG(fs.TotalRevenue) as AvgOrderValue
            FROM FactSales fs
            JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
            JOIN DimDate d ON fs.DateKey = d.DateKey
            WHERE c.CustomerID != 'UNKNOWN'
            GROUP BY c.CustomerKey, c.CustomerID, c.Country, c.CustomerSegment
        ),
        rfm_scores AS (
            SELECT *,
                CASE 
                    WHEN LastPurchaseDate >= date('now', '-30 days') THEN 5
                    WHEN LastPurchaseDate >= date('now', '-60 days') THEN 4
                    WHEN LastPurchaseDate >= date('now', '-90 days') THEN 3
                    WHEN LastPurchaseDate >= date('now', '-180 days') THEN 2
                    ELSE 1
                END as RecencyScore,
                CASE 
                    WHEN Frequency >= 50 THEN 5
                    WHEN Frequency >= 20 THEN 4
                    WHEN Frequency >= 10 THEN 3
                    WHEN Frequency >= 5 THEN 2
                    ELSE 1
                END as FrequencyScore,
                CASE 
                    WHEN Monetary >= 5000 THEN 5
                    WHEN Monetary >= 2000 THEN 4
                    WHEN Monetary >= 1000 THEN 3
                    WHEN Monetary >= 500 THEN 2
                    ELSE 1
                END as MonetaryScore
            FROM customer_metrics
        )
        SELECT *,
            (RecencyScore + FrequencyScore + MonetaryScore) as RFMScore,
            CASE 
                WHEN RecencyScore >= 4 AND FrequencyScore >= 4 AND MonetaryScore >= 4 THEN 'Champions'
                WHEN RecencyScore >= 3 AND FrequencyScore >= 3 AND MonetaryScore >= 3 THEN 'Loyal Customers'
                WHEN RecencyScore >= 4 AND FrequencyScore <= 2 AND MonetaryScore >= 3 THEN 'Potential Loyalists'
                WHEN RecencyScore >= 3 AND FrequencyScore >= 2 AND MonetaryScore <= 2 THEN 'New Customers'
                WHEN RecencyScore >= 3 AND FrequencyScore <= 2 AND MonetaryScore <= 2 THEN 'Promising'
                WHEN RecencyScore <= 2 AND FrequencyScore >= 3 AND MonetaryScore >= 3 THEN 'Need Attention'
                WHEN RecencyScore <= 2 AND FrequencyScore >= 2 AND MonetaryScore >= 2 THEN 'About to Sleep'
                WHEN RecencyScore <= 2 AND FrequencyScore <= 2 AND MonetaryScore >= 3 THEN 'At Risk'
                WHEN RecencyScore <= 2 AND FrequencyScore <= 2 AND MonetaryScore <= 2 THEN 'Cannot Lose Them'
                ELSE 'Others'
            END as RFMSegment
        FROM rfm_scores
        ORDER BY RFMScore DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"RFM Analysis completed: {len(df)} customers analyzed")
            return df
        except Exception as e:
            self.logger.error(f"Error in RFM analysis: {e}")
            return pd.DataFrame()
    
    def get_customer_lifetime_value(self) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV)
        
        Returns:
            DataFrame with CLV metrics
        """
        query = """
        WITH customer_clv AS (
            SELECT 
                c.CustomerKey,
                c.CustomerID,
                c.Country,
                COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
                SUM(fs.TotalRevenue) as TotalRevenue,
                AVG(fs.TotalRevenue) as AvgOrderValue,
                MIN(d.FullDate) as FirstPurchaseDate,
                MAX(d.FullDate) as LastPurchaseDate,
                (julianday(MAX(d.FullDate)) - julianday(MIN(d.FullDate))) / 365.0 as CustomerLifespanYears,
                COUNT(DISTINCT d.FullDate) as ActiveDays
            FROM FactSales fs
            JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
            JOIN DimDate d ON fs.DateKey = d.DateKey
            WHERE c.CustomerID != 'UNKNOWN'
            GROUP BY c.CustomerKey, c.CustomerID, c.Country
        )
        SELECT *,
            CASE 
                WHEN CustomerLifespanYears > 0 THEN TotalRevenue / CustomerLifespanYears
                ELSE TotalRevenue
            END as AnnualRevenue,
            CASE 
                WHEN ActiveDays > 0 THEN TotalOrders / (ActiveDays / 30.0)
                ELSE 0
            END as MonthlyOrderFrequency,
            CASE 
                WHEN TotalRevenue >= 10000 THEN 'Premium'
                WHEN TotalRevenue >= 5000 THEN 'Gold'
                WHEN TotalRevenue >= 2000 THEN 'Silver'
                WHEN TotalRevenue >= 1000 THEN 'Bronze'
                ELSE 'Basic'
            END as CustomerTier
        FROM customer_clv
        ORDER BY TotalRevenue DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"CLV Analysis completed: {len(df)} customers analyzed")
            return df
        except Exception as e:
            self.logger.error(f"Error in CLV analysis: {e}")
            return pd.DataFrame()
    
    def get_product_performance_analysis(self) -> pd.DataFrame:
        """
        Comprehensive product performance analysis with improved metrics
        
        Returns:
            DataFrame with product metrics
        """
        query = """
        WITH product_metrics AS (
            SELECT 
                p.ProductKey,
                p.StockCode,
                p.Description,
                p.ProductCategory as OriginalCategory,
                COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
                SUM(fs.Quantity) as TotalQuantity,
                SUM(fs.TotalRevenue) as TotalRevenue,
                AVG(fs.UnitPrice) as AvgUnitPrice,
                COUNT(DISTINCT fs.CustomerKey) as UniqueCustomers,
                COUNT(DISTINCT d.FullDate) as ActiveDays,
                MIN(d.FullDate) as FirstSaleDate,
                MAX(d.FullDate) as LastSaleDate
            FROM FactSales fs
            JOIN DimProduct p ON fs.ProductKey = p.ProductKey
            JOIN DimDate d ON fs.DateKey = d.DateKey
            GROUP BY p.ProductKey, p.StockCode, p.Description, p.ProductCategory
        )
        SELECT *,
            -- Improved monthly calculations
            CASE 
                WHEN ActiveDays > 0 AND ActiveDays < 365 THEN TotalQuantity * 30.0 / ActiveDays
                WHEN ActiveDays >= 365 THEN TotalQuantity * 30.0 / 365.0
                ELSE TotalQuantity
            END as MonthlyQuantity,
            CASE 
                WHEN ActiveDays > 0 AND ActiveDays < 365 THEN TotalRevenue * 30.0 / ActiveDays
                WHEN ActiveDays >= 365 THEN TotalRevenue * 30.0 / 365.0
                ELSE TotalRevenue
            END as MonthlyRevenue,
            -- Additional metrics
            CASE 
                WHEN UniqueCustomers > 0 THEN TotalOrders / UniqueCustomers
                ELSE 0
            END as OrdersPerCustomer,
            CASE 
                WHEN TotalQuantity > 0 THEN TotalRevenue / TotalQuantity
                ELSE 0
            END as RevenuePerUnit,
            CASE 
                WHEN ActiveDays > 0 THEN TotalRevenue / ActiveDays
                ELSE 0
            END as DailyRevenue,
            -- Product lifecycle stage
            CASE 
                WHEN ActiveDays <= 30 THEN 'New'
                WHEN ActiveDays <= 90 THEN 'Growing'
                WHEN ActiveDays <= 365 THEN 'Mature'
                ELSE 'Established'
            END as LifecycleStage,
            -- Product performance category based on revenue and frequency
            CASE 
                WHEN TotalRevenue >= 50000 AND TotalOrders >= 100 THEN 'Star'
                WHEN TotalRevenue >= 20000 AND TotalOrders >= 50 THEN 'Cash Cow'
                WHEN TotalRevenue >= 10000 AND TotalOrders >= 20 THEN 'Question Mark'
                WHEN TotalRevenue >= 5000 THEN 'Potential'
                ELSE 'Dog'
            END as ProductCategory
        FROM product_metrics
        WHERE TotalRevenue > 0  -- Only include products with sales
        ORDER BY TotalRevenue DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Product Performance Analysis completed: {len(df)} products analyzed")
            return df
        except Exception as e:
            self.logger.error(f"Error in product performance analysis: {e}")
            return pd.DataFrame()
    
    def get_seasonal_analysis(self) -> pd.DataFrame:
        """
        Seasonal trend analysis
        
        Returns:
            DataFrame with seasonal metrics
        """
        query = """
        SELECT 
            d.Year,
            d.Month,
            d.MonthName,
            d.Quarter,
            COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
            SUM(fs.Quantity) as TotalQuantity,
            SUM(fs.TotalRevenue) as TotalRevenue,
            AVG(fs.TotalRevenue) as AvgOrderValue,
            COUNT(DISTINCT fs.CustomerKey) as UniqueCustomers,
            COUNT(DISTINCT fs.ProductKey) as UniqueProducts
        FROM FactSales fs
        JOIN DimDate d ON fs.DateKey = d.DateKey
        GROUP BY d.Year, d.Month, d.MonthName, d.Quarter
        ORDER BY d.Year, d.Month
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Seasonal Analysis completed: {len(df)} periods analyzed")
            return df
        except Exception as e:
            self.logger.error(f"Error in seasonal analysis: {e}")
            return pd.DataFrame()
    
    def get_geographic_analysis(self) -> pd.DataFrame:
        """
        Geographic performance analysis
        
        Returns:
            DataFrame with geographic metrics
        """
        query = """
        SELECT 
            l.Country,
            l.Region,
            l.Continent,
            COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
            SUM(fs.Quantity) as TotalQuantity,
            SUM(fs.TotalRevenue) as TotalRevenue,
            AVG(fs.TotalRevenue) as AvgOrderValue,
            COUNT(DISTINCT fs.CustomerKey) as UniqueCustomers,
            COUNT(DISTINCT fs.ProductKey) as UniqueProducts,
            CASE 
                WHEN SUM(fs.TotalRevenue) >= 100000 THEN 'Major Market'
                WHEN SUM(fs.TotalRevenue) >= 50000 THEN 'Important Market'
                WHEN SUM(fs.TotalRevenue) >= 10000 THEN 'Growing Market'
                ELSE 'Emerging Market'
            END as MarketSegment
        FROM FactSales fs
        JOIN DimLocation l ON fs.LocationKey = l.LocationKey
        GROUP BY l.Country, l.Region, l.Continent
        ORDER BY TotalRevenue DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Geographic Analysis completed: {len(df)} locations analyzed")
            return df
        except Exception as e:
            self.logger.error(f"Error in geographic analysis: {e}")
            return pd.DataFrame()
    
    def get_cohort_analysis(self) -> pd.DataFrame:
        """
        Customer cohort analysis
        
        Returns:
            DataFrame with cohort metrics
        """
        query = """
        WITH customer_cohorts AS (
            SELECT 
                c.CustomerKey,
                c.CustomerID,
                MIN(d.FullDate) as FirstPurchaseDate,
                strftime('%Y-%m', MIN(d.FullDate)) as CohortMonth
            FROM FactSales fs
            JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
            JOIN DimDate d ON fs.DateKey = d.DateKey
            WHERE c.CustomerID != 'UNKNOWN'
            GROUP BY c.CustomerKey, c.CustomerID
        ),
        cohort_periods AS (
            SELECT 
                fs.CustomerKey,
                cc.CohortMonth,
                strftime('%Y-%m', d.FullDate) as OrderMonth,
                (strftime('%Y', d.FullDate) - strftime('%Y', cc.FirstPurchaseDate)) * 12 + 
                (strftime('%m', d.FullDate) - strftime('%m', cc.FirstPurchaseDate)) as PeriodNumber
            FROM FactSales fs
            JOIN DimDate d ON fs.DateKey = d.DateKey
            JOIN customer_cohorts cc ON fs.CustomerKey = cc.CustomerKey
        )
        SELECT 
            CohortMonth,
            PeriodNumber,
            COUNT(DISTINCT CustomerKey) as Customers,
            COUNT(DISTINCT CASE WHEN PeriodNumber = 0 THEN CustomerKey END) as CohortSize
        FROM cohort_periods
        GROUP BY CohortMonth, PeriodNumber
        HAVING CohortSize > 0
        ORDER BY CohortMonth, PeriodNumber
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Cohort Analysis completed: {len(df)} cohort periods analyzed")
            return df
        except Exception as e:
            self.logger.error(f"Error in cohort analysis: {e}")
            return pd.DataFrame()
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights report
        
        Returns:
            Dictionary with key insights
        """
        insights = {}
        
        try:
            # RFM Analysis
            rfm_df = self.get_rfm_analysis()
            if not rfm_df.empty:
                insights['rfm'] = {
                    'total_customers': len(rfm_df),
                    'champions': len(rfm_df[rfm_df['RFMSegment'] == 'Champions']),
                    'at_risk': len(rfm_df[rfm_df['RFMSegment'] == 'At Risk']),
                    'new_customers': len(rfm_df[rfm_df['RFMSegment'] == 'New Customers']),
                    'avg_rfm_score': rfm_df['RFMScore'].mean()
                }
            
            # CLV Analysis
            clv_df = self.get_customer_lifetime_value()
            if not clv_df.empty:
                insights['clv'] = {
                    'avg_clv': clv_df['TotalRevenue'].mean(),
                    'premium_customers': len(clv_df[clv_df['CustomerTier'] == 'Premium']),
                    'avg_order_frequency': clv_df['MonthlyOrderFrequency'].mean()
                }
            
            # Product Performance
            product_df = self.get_product_performance_analysis()
            if not product_df.empty:
                insights['products'] = {
                    'total_products': len(product_df),
                    'star_products': len(product_df[product_df['ProductCategory'] == 'Star']),
                    'top_product_revenue': product_df['TotalRevenue'].max(),
                    'avg_product_revenue': product_df['TotalRevenue'].mean()
                }
            
            # Geographic Analysis
            geo_df = self.get_geographic_analysis()
            if not geo_df.empty:
                insights['geography'] = {
                    'total_countries': len(geo_df),
                    'major_markets': len(geo_df[geo_df['MarketSegment'] == 'Major Market']),
                    'top_country': geo_df.iloc[0]['Country'] if len(geo_df) > 0 else None,
                    'top_country_revenue': geo_df['TotalRevenue'].max()
                }
            
            self.logger.info("Insights report generated successfully")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights report: {e}")
            return {}

def main():
    """Main function to demonstrate advanced analytics"""
    analytics = AdvancedAnalytics()
    
    print("=== Advanced Analytics Demo ===")
    
    # RFM Analysis
    print("\n1. RFM Analysis:")
    rfm_df = analytics.get_rfm_analysis()
    if not rfm_df.empty:
        print(f"   Total customers analyzed: {len(rfm_df)}")
        print(f"   Champions: {len(rfm_df[rfm_df['RFMSegment'] == 'Champions'])}")
        print(f"   At Risk: {len(rfm_df[rfm_df['RFMSegment'] == 'At Risk'])}")
    
    # CLV Analysis
    print("\n2. Customer Lifetime Value:")
    clv_df = analytics.get_customer_lifetime_value()
    if not clv_df.empty:
        print(f"   Average CLV: ${clv_df['TotalRevenue'].mean():,.2f}")
        print(f"   Premium customers: {len(clv_df[clv_df['CustomerTier'] == 'Premium'])}")
    
    # Product Performance
    print("\n3. Product Performance:")
    product_df = analytics.get_product_performance_analysis()
    if not product_df.empty:
        print(f"   Total products: {len(product_df)}")
        print(f"   Star products: {len(product_df[product_df['ProductCategory'] == 'Star'])}")
    
    # Generate insights
    print("\n4. Key Insights:")
    insights = analytics.generate_insights_report()
    for category, data in insights.items():
        print(f"   {category.title()}: {data}")

if __name__ == "__main__":
    main()

