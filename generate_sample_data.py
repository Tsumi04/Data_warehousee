"""
Generate Sample Online Retail Data for testing
This creates a realistic sample dataset based on the Online Retail structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Sample product data
PRODUCTS = [
    ("PEN", "PEN", 5.00),
    ("PENS", "PENS", 2.50),
    ("WHITE HANGING HEART T-LIGHT HOLDER", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("CREAM CUPID HEARTS COAT HANGER", "CREAM CUPID HEARTS COAT HANGER", 4.95),
    ("FELTCRAFT PRINCESS CHARLOTTE DOLL", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75),
    ("RED WOOLLY HOTTIE WHITE HEART.", "RED WOOLLY HOTTIE WHITE HEART.", 3.39),
    ("DOORMAT NEW ENGLAND", "DOORMAT NEW ENGLAND", 7.95),
    ("SET 7 BABUSHKA NESTING BOXES", "SET 7 BABUSHKA NESTING BOXES", 5.95),
    ("POPCORN HOLDER", "POPCORN HOLDER", 1.69),
    ("PANTRY SCRAPBOOK SET", "PANTRY SCRAPBOOK SET", 8.95),
    ("METAL SIGN", "METAL SIGN", 4.95),
    ("VINTAGE CARAVAN KEY RING", "VINTAGE CARAVAN KEY RING", 4.13),
    ("HAND WARMER BABUSHKA DESIGN", "HAND WARMER BABUSHKA DESIGN", 6.95),
    ("CHILDRENS CUTLERY CIRCUS PARADE", "CHILDRENS CUTLERY CIRCUS PARADE", 4.21),
    ("BLACK RECORD COVER FRAMES", "BLACK RECORD COVER FRAMES", 12.75),
]

COUNTRIES = [
    'United Kingdom', 'France', 'Germany', 'Australia', 'Spain', 
    'Italy', 'Netherlands', 'Norway', 'Belgium', 'Sweden',
    'Switzerland', 'Portugal', 'Denmark', 'Finland', 'Austria'
]

def generate_sample_data(n_records=10000):
    """
    Generate sample retail transaction data.
    
    Args:
        n_records: Number of records to generate
    
    Returns:
        pandas.DataFrame
    """
    print(f"Generating {n_records:,} sample records...")
    
    data = []
    
    # Date range: 2010-12-01 to 2011-12-09 (matching original dataset period)
    start_date = datetime(2010, 12, 1)
    end_date = datetime(2011, 12, 9)
    
    # Generate some customer IDs
    customer_ids = [f"{int(np.random.uniform(12347, 18287))}" for _ in range(n_records)]
    # Add some nulls
    customer_ids = [None if random.random() < 0.05 else cid for cid in customer_ids]
    
    # Generate invoice numbers (some cancellations)
    invoice_nos = []
    for i in range(n_records):
        invoice_no = f"{int(np.random.uniform(536000, 581500))}"
        if random.random() < 0.02:  # 2% cancellations
            invoice_no = 'C' + invoice_no
        invoice_nos.append(invoice_no)
    
    # Generate transactions
    for i in range(n_records):
        # Random date
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        invoice_date = start_date + timedelta(days=random_days, hours=random.randint(8, 20))
        
        # Random product
        stock_code, description, unit_price = random.choice(PRODUCTS)
        
        # Quantity (mostly positive, some negatives for returns)
        quantity = max(1, int(np.random.exponential(3)))
        if random.random() < 0.05:  # 5% returns
            quantity = -quantity
        
        # Skip if cancelled and quantity is negative
        if invoice_nos[i].startswith('C'):
            quantity = -abs(quantity)
        
        country = random.choice(COUNTRIES)
        customer_id = customer_ids[i]
        
        data.append({
            'InvoiceNo': invoice_nos[i],
            'StockCode': stock_code,
            'Description': description,
            'Quantity': quantity,
            'InvoiceDate': invoice_date,
            'UnitPrice': unit_price,
            'CustomerID': customer_id,
            'Country': country
        })
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1:,} / {n_records:,} records...")
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Generated {len(df):,} sample records successfully!")
    
    return df


def main():
    """Main function to generate and save sample data."""
    print("="*80)
    print("GENERATING SAMPLE RETAIL DATA")
    print("="*80)
    
    # Generate 10,000 records (good size for demo)
    df = generate_sample_data(n_records=10000)
    
    # Save to file
    output_path = 'data/raw/online_retail.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nSample data saved to: {output_path}")
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"   Countries: {df['Country'].nunique()}")
    print(f"   Products: {df['StockCode'].nunique()}")
    print(f"   Customers: {df['CustomerID'].notna().sum():,}")
    
    print("\nYou can now run: python run_all.py")
    print("="*80)


if __name__ == '__main__':
    main()

