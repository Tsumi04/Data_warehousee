"""
Script to start the Streamlit dashboard
"""

import subprocess
import sys
import webbrowser
import time
import os

def start_dashboard():
    """Start the Streamlit dashboard"""
    
    print("="*80)
    print("STARTING RETAIL ANALYTICS DASHBOARD")
    print("="*80)
    
    # Get the path to this directory
    app_path = os.path.join(os.getcwd(), "app.py")
    
    print(f"Dashboard file: {app_path}")
    print(f"Opening browser in 5 seconds...")
    print(f"URL: http://localhost:8501")
    print("="*80)
    
    # Start Streamlit in background
    print("\nStarting Streamlit server...")
    print("Press Ctrl+C to stop the server")
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        print("\nTrying alternative method...")
        
        # Alternative: just print instructions
        print("\n" + "="*80)
        print("MANUAL START INSTRUCTIONS:")
        print("="*80)
        print("\n1. Open a new terminal/command prompt")
        print("2. Navigate to this directory:")
        print(f"   cd {os.getcwd()}")
        print("\n3. Run:")
        print("   python -m streamlit run app.py")
        print("\n4. Your browser will open automatically to:")
        print("   http://localhost:8501")
        print("\n" + "="*80)


if __name__ == '__main__':
    start_dashboard()


