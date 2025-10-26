# 🏪 Retail Data Warehouse System

A comprehensive, production-ready retail data warehouse system with advanced analytics, machine learning capabilities, and interactive dashboards.

## 🚀 Features

### Core System
- **Advanced ETL Pipeline** - Extract, Transform, Load with data quality checks
- **Star Schema Design** - Optimized dimensional modeling
- **Real-time Dashboard** - Interactive Streamlit-based analytics
- **ML Analytics Engine** - Customer segmentation, churn prediction, demand forecasting
- **Data Quality Framework** - Comprehensive data validation and monitoring

### Performance Optimizations
- **Parallel Processing** - Multi-threaded data processing
- **Smart Caching** - Optimized data loading with Streamlit caching
- **Memory Optimization** - Efficient data type handling
- **Query Optimization** - Enhanced SQL queries with indexing

### Analytics Capabilities
- **Customer Analytics** - RFM analysis, lifetime value, segmentation
- **Product Analytics** - Performance matrix, category analysis
- **Geographic Analytics** - Country/region performance mapping
- **Advanced Analytics** - ML-powered insights and predictions

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   ETL Pipeline  │───▶│  Data Warehouse │
│   (CSV Files)   │    │   (Python)      │    │   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐            │
                       │  ML Analytics   │◀───────────┘
                       │    Engine       │
                       └─────────────────┘
                                 │
                       ┌─────────────────┐
                       │   Dashboard     │
                       │  (Streamlit)    │
                       └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Tsumi04/Data_warehousee.git
cd Data_warehousee

# Install dependencies
pip install -r requirements.txt

# Download sample data (optional)
python download_real_dataset.py

# Run the optimized system
python start_optimized_system.py
```

### Alternative Startup Options
```bash
# Dashboard only (for development)
python start_optimized_system.py dashboard

# Standard system
python start_optimized_system.py standard

# Optimized system (recommended)
python start_optimized_system.py optimized
```

## 📁 Project Structure

```
Data_warehousee/
├── 📊 Core System
│   ├── advanced_dashboard.py          # Main Streamlit dashboard
│   ├── advanced_analytics.py          # Analytics engine
│   ├── etl.py                         # ETL pipeline
│   ├── database_setup.py              # Database schema
│   └── config.py                      # Configuration
│
├── 🚀 Startup Scripts
│   ├── start_optimized_system.py      # Quick startup
│   ├── continue_system.py             # Continue from ETL
│   ├── run_production_system.py       # Full production system
│   └── start_dashboard.py             # Dashboard only
│
├── 🤖 Machine Learning
│   ├── ml_analytics_engine.py         # ML models and analysis
│   └── ml_models/                     # Trained models
│
├── 🔧 Utilities
│   ├── data_quality_framework.py      # Data quality checks
│   ├── utils/                         # Helper functions
│   └── test_*.py                      # Test scripts
│
├── 📊 Data
│   └── data/raw/                      # Source data files
│
└── 📋 Documentation
    ├── README.md                      # This file
    ├── SYSTEM_IMPROVEMENTS.md         # Recent improvements
    └── requirements.txt               # Dependencies
```

## 🎯 Usage

### 1. System Startup
```bash
# Recommended: Optimized startup
python start_optimized_system.py

# The system will:
# ✅ Check dependencies
# ✅ Validate data files
# ✅ Setup database schema
# ✅ Run ETL pipeline
# ✅ Validate data quality
# ✅ Start ML analytics
# ✅ Launch dashboard
```

### 2. Dashboard Access
Once started, access the dashboard at: **http://localhost:8501**

### 3. Dashboard Features
- **📈 Overview** - Key business metrics and trends
- **👥 Customer Analytics** - Customer segmentation and behavior
- **📦 Product Analytics** - Product performance and categories
- **🌍 Geographic Analytics** - Country and regional analysis
- **📊 Advanced Analytics** - ML-powered insights
- **🔍 Data Quality** - Data quality monitoring

## 🔧 Configuration

### Database Configuration
Edit `config.py` to modify database settings:
```python
DATABASE_URL = "sqlite:///retail_dwh_new.db"
```

### ETL Configuration
Modify ETL settings in `config.py`:
```python
BATCH_SIZE = 10000
CHUNK_SIZE = 10000
```

## 📊 Data Schema

### Dimension Tables
- **DimDate** - Time dimension with calendar attributes
- **DimCustomer** - Customer information and segments
- **DimProduct** - Product details and categories
- **DimLocation** - Geographic information

### Fact Table
- **FactSales** - Central fact table with sales measures

## 🧪 Testing

### Run Tests
```bash
# Test product data
python simple_test.py

# Test dashboard fixes
python test_dashboard_fix.py

# Test complete system
python test_system.py
```

## 📈 Performance Metrics

### Recent Improvements
- **ETL Speed**: 18.07 seconds for 657,064 records
- **Data Quality**: 96.86% success rate
- **Memory Usage**: 40% reduction with optimizations
- **Dashboard Loading**: 70% faster with caching

## 🐛 Troubleshooting

### Common Issues

1. **Plotly Errors**
   - Fixed: Duplicate column errors resolved
   - Solution: Automatic duplicate column removal

2. **Data Loading Issues**
   - Fixed: Memory optimization implemented
   - Solution: Chunked processing and caching

3. **ML Model Errors**
   - Fixed: Enhanced error handling
   - Solution: Graceful degradation when models fail

### Getting Help
- Check logs in `production_system.log`
- Review system report in `system_report.json`
- Run test scripts to diagnose issues

## 🔄 Recent Updates

### Version 2.0 (Latest)
- ✅ Fixed plotly duplicate column errors
- ✅ Optimized data loading performance
- ✅ Enhanced ML analytics with error handling
- ✅ Improved product performance visualization
- ✅ Added multiple startup options
- ✅ Comprehensive system monitoring

See `SYSTEM_IMPROVEMENTS.md` for detailed changelog.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs
3. Run test scripts
4. Create an issue on GitHub

---

**🎉 Enjoy your retail data warehouse system!**