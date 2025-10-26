#!/usr/bin/env python3
"""
Simple test for product data
"""

import pandas as pd
import sys

def test_product_data():
    """Test product performance data"""
    print("Testing product performance data...")
    
    try:
        from advanced_analytics import AdvancedAnalytics
        
        # Initialize analytics
        analytics = AdvancedAnalytics()
        
        # Test product performance analysis
        print("Getting product performance data...")
        product_df = analytics.get_product_performance_analysis()
        
        if not product_df.empty:
            print(f"SUCCESS: {len(product_df)} products found")
            
            # Check data ranges
            print(f"MonthlyQuantity - Min: {product_df['MonthlyQuantity'].min():.2f}, Max: {product_df['MonthlyQuantity'].max():.2f}")
            print(f"MonthlyRevenue - Min: {product_df['MonthlyRevenue'].min():.2f}, Max: {product_df['MonthlyRevenue'].max():.2f}")
            print(f"TotalRevenue - Min: {product_df['TotalRevenue'].min():.2f}, Max: {product_df['TotalRevenue'].max():.2f}")
            
            # Check product categories
            print(f"Product Categories:")
            category_counts = product_df['ProductCategory'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count} products")
            
            # Show top 5 products
            print(f"Top 5 Products by Revenue:")
            top_products = product_df.head(5)[['StockCode', 'Description', 'TotalRevenue', 'MonthlyRevenue', 'ProductCategory']]
            for idx, row in top_products.iterrows():
                desc = row['Description'][:30] + "..." if len(row['Description']) > 30 else row['Description']
                print(f"  {row['StockCode']}: {desc} - ${row['TotalRevenue']:,.2f} (${row['MonthlyRevenue']:,.2f}/month) - {row['ProductCategory']}")
            
            return True
        else:
            print("ERROR: Product analysis returned empty DataFrame")
            return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING PRODUCT PERFORMANCE DATA")
    print("=" * 50)
    
    success = test_product_data()
    
    if success:
        print("\nSUCCESS: Product data test completed!")
    else:
        print("\nFAILED: Product data test failed.")
        sys.exit(1)
