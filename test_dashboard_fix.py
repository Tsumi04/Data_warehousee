#!/usr/bin/env python3
"""
Test script to verify the dashboard duplicate column fix
"""

import pandas as pd
import sys
from pathlib import Path

def test_duplicate_column_fix():
    """Test the duplicate column fix"""
    print("Testing duplicate column fix...")
    
    try:
        from advanced_analytics import AdvancedAnalytics
        from config import DATABASE_URL
        
        # Initialize analytics
        analytics = AdvancedAnalytics()
        
        # Test product performance analysis
        print("Testing product performance analysis...")
        product_df = analytics.get_product_performance_analysis()
        
        if not product_df.empty:
            print(f"✅ Product analysis successful: {len(product_df)} products")
            print(f"Columns: {list(product_df.columns)}")
            
            # Check for duplicate columns
            duplicate_cols = product_df.columns[product_df.columns.duplicated()].tolist()
            if duplicate_cols:
                print(f"❌ Found duplicate columns: {duplicate_cols}")
                return False
            else:
                print("✅ No duplicate columns found")
            
            # Test the duplicate column removal
            product_df_clean = product_df.loc[:, ~product_df.columns.duplicated()]
            print(f"✅ Duplicate column removal works: {len(product_df_clean.columns)} columns")
            
        else:
            print("❌ Product analysis returned empty DataFrame")
            return False
        
        # Test RFM analysis
        print("\nTesting RFM analysis...")
        rfm_df = analytics.get_rfm_analysis()
        
        if not rfm_df.empty:
            print(f"✅ RFM analysis successful: {len(rfm_df)} customers")
            duplicate_cols = rfm_df.columns[rfm_df.columns.duplicated()].tolist()
            if duplicate_cols:
                print(f"❌ Found duplicate columns in RFM: {duplicate_cols}")
                return False
            else:
                print("✅ No duplicate columns in RFM data")
        else:
            print("❌ RFM analysis returned empty DataFrame")
            return False
        
        # Test CLV analysis
        print("\nTesting CLV analysis...")
        clv_df = analytics.get_customer_lifetime_value()
        
        if not clv_df.empty:
            print(f"✅ CLV analysis successful: {len(clv_df)} customers")
            duplicate_cols = clv_df.columns[clv_df.columns.duplicated()].tolist()
            if duplicate_cols:
                print(f"❌ Found duplicate columns in CLV: {duplicate_cols}")
                return False
            else:
                print("✅ No duplicate columns in CLV data")
        else:
            print("❌ CLV analysis returned empty DataFrame")
            return False
        
        print("\n🎉 All tests passed! Dashboard should work without duplicate column errors.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 TESTING DASHBOARD DUPLICATE COLUMN FIX")
    print("=" * 60)
    
    success = test_duplicate_column_fix()
    
    if success:
        print("\n✅ All tests passed! The dashboard should now work without errors.")
        print("You can now run: python continue_system.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
