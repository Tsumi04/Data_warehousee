#!/usr/bin/env python3
"""
Simple script to check database status
"""

import sqlite3
import os

def check_database():
    """Check database tables and data"""
    db_file = "retail_dwh_new.db"
    
    if not os.path.exists(db_file):
        print(f"Database file {db_file} not found")
        return
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("Tables in database:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check data in each table
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count} records")
        
        conn.close()
        print("Database check completed successfully")
        
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_database()


