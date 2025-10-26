#!/usr/bin/env python3
"""
Test script to check product performance data
"""

import pandas as pd
import sys
from pathlib import Path

def test_product_data():
    """Test product performance data"""
    print("Testing product performance data...")
    
    try:
        from advanced_analytics import AdvancedAnalytics
        from config import DATABASE_URL
        
        # Initialize analytics
        analytics = AdvancedAnalytics()
        
        # Test product performance analysis
        print("Getting product performance data...")
        product_df = analytics.get_product_performance_analysis()
        
        if not product_df.empty:
            print(f"✅ Product analysis successful: {len(product_df)} products")
            print(f"Columns: {list(product_df.columns)}")
            
            # Check data ranges
            print("\n📊 Data Statistics:")
            print(f"MonthlyQuantity - Min: {product_df['MonthlyQuantity'].min():.2f}, Max: {product_df['MonthlyQuantity'].max():.2f}, Mean: {product_df['MonthlyQuantity'].mean():.2f}")
            print(f"MonthlyRevenue - Min: {product_df['MonthlyRevenue'].min():.2f}, Max: {product_df['MonthlyRevenue'].max():.2f}, Mean: {product_df['MonthlyRevenue'].mean():.2f}")
            print(f"TotalRevenue - Min: {product_df['TotalRevenue'].min():.2f}, Max: {product_df['TotalRevenue'].max():.2f}, Mean: {product_df['TotalRevenue'].mean():.2f}")
            
            # Check product categories
            print(f"\n📦 Product Categories:")
            category_counts = product_df['ProductCategory'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count} products")
            
            # Check lifecycle stages
            print(f"\n🔄 Lifecycle Stages:")
            lifecycle_counts = product_df['LifecycleStage'].value_counts()
            for stage, count in lifecycle_counts.items():
                print(f"  {stage}: {count} products")
            
            # Show top products
            print(f"\n🏆 Top 10 Products by Revenue:")
            top_products = product_df.head(10)[['StockCode', 'Description', 'TotalRevenue', 'MonthlyRevenue', 'ProductCategory']]
            for idx, row in top_products.iterrows():
                print(f"  {row['StockCode']}: {row['Description'][:50]}... - ${row['TotalRevenue']:,.2f} (${row['MonthlyRevenue']:,.2f}/month) - {row['ProductCategory']}")
            
            # Check for zero values
            zero_quantity = len(product_df[product_df['MonthlyQuantity'] == 0])
            zero_revenue = len(product_df[product_df['MonthlyRevenue'] == 0])
            print(f"\n⚠️  Products with zero values:")
            print(f"  Zero MonthlyQuantity: {zero_quantity}")
            print(f"  Zero MonthlyRevenue: {zero_revenue}")
            
            # Filter for meaningful data
            meaningful_data = product_df[
                (product_df['MonthlyQuantity'] > 0) | 
                (product_df['MonthlyRevenue'] > 0)
            ]
            print(f"\n📈 Meaningful data points: {len(meaningful_data)} out of {len(product_df)}")
            
            if len(meaningful_data) > 0:
                print(f"Meaningful data ranges:")
                print(f"  MonthlyQuantity - Min: {meaningful_data['MonthlyQuantity'].min():.2f}, Max: {meaningful_data['MonthlyQuantity'].max():.2f}")
                print(f"  MonthlyRevenue - Min: {meaningful_data['MonthlyRevenue'].min():.2f}, Max: {meaningful_data['MonthlyRevenue'].max():.2f}")
            
            return True
        else:
            print("❌ Product analysis returned empty DataFrame")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING PRODUCT PERFORMANCE DATA")
    print("=" * 60)
    
    success = test_product_data()
    
    if success:
        print("\n✅ Product data test completed!")
        print("The dashboard should now show meaningful data.")
    else:
        print("\n❌ Product data test failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
