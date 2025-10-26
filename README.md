# 📊 Retail Data Warehouse & Analytics Dashboard

Một dự án Data Warehouse hoàn chỉnh và production-ready cho phân tích dữ liệu bán lẻ, sử dụng Python, SQLAlchemy, và Streamlit.

## 🎯 Tổng quan Dự án

Dự án này thiết kế và xây dựng một giải pháp kho dữ liệu (Data Warehouse) và dashboard Business Intelligence (BI) hoàn chỉnh cho lĩnh vực bán lẻ. Dự án áp dụng các nguyên tắc kiến trúc dữ liệu vững chắc để tạo ra một hệ thống phân tích đáng tin cậy, hiệu quả và có khả năng mở rộng.

### Đặc điểm chính

- ✅ **Star Schema Architecture**: Mô hình dữ liệu đa chiều tối ưu cho truy vấn BI
- ✅ **ETL Pipeline hoàn chỉnh**: Xử lý dữ liệu thô với các bước Extract, Transform, Load
- ✅ **Data Quality Assurance**: Xử lý missing data, outliers, và validation
- ✅ **Interactive Dashboard**: Streamlit dashboard với nhiều visualization và filters
- ✅ **Production-Ready**: Code được chuẩn hóa, có logging, error handling

## 📁 Cấu trúc Dự án

```
Data_warehousee/
├── README.md                  # Tài liệu dự án
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
├── config.py                 # Configuration settings
├── database_setup.py         # Database schema và models
├── etl.py                    # ETL pipeline chính
├── app.py                    # Streamlit dashboard
├── data/                     # Thư mục dữ liệu
│   ├── raw/                  # Dữ liệu thô
│   ├── clean/                # Dữ liệu đã làm sạch
│   └── processed/            # Dữ liệu đã xử lý
└── retail_dwh.db             # SQLite database (sẽ được tạo)
```

## 🗄️ Data Warehouse Schema

Dự án sử dụng **Star Schema** với các bảng sau:

### Dimension Tables (Bảng Chiều)

1. **DimDate**: Chiều thời gian
   - DateKey (PK)
   - FullDate, Day, Month, Year, Quarter
   - DayName, MonthName, IsWeekend, IsHoliday

2. **DimCustomer**: Chiều khách hàng
   - CustomerKey (PK)
   - CustomerID, Country, CustomerSegment

3. **DimProduct**: Chiều sản phẩm
   - ProductKey (PK)
   - StockCode, Description, ProductCategory

4. **DimLocation**: Chiều địa lý
   - LocationKey (PK)
   - Country, Region, Continent

### Fact Table (Bảng Sự kiện)

**FactSales**: Bảng trung tâm chứa các số đo kinh doanh
- SalesKey (PK)
- DateKey, CustomerKey, ProductKey, LocationKey (Foreign Keys)
- InvoiceNo, Quantity, UnitPrice, TotalRevenue
- LoadTimestamp

## 🚀 Hướng dẫn Cài đặt & Chạy

### Yêu cầu hệ thống

- Python 3.8+
- pip package manager

### Bước 1: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Tải Dataset

Tải dataset "Online Retail" từ Kaggle:
https://www.kaggle.com/datasets/thedevastator/online-retail-sales-and-customer-data

Hoặc sử dụng Kaggle API:
```bash
pip install kaggle
# Đặt kaggle.json vào ~/.kaggle/
kaggle datasets download -d thedevastator/online-retail-sales-and-customer-data
```

Đặt file CSV vào thư mục: `data/raw/online_retail.csv`

### Bước 3: Tạo Database

```bash
python database_setup.py
```

Lệnh này sẽ tạo file SQLite `retail_dwh.db` và các bảng cần thiết.

### Bước 4: Chạy ETL Pipeline

```bash
python etl.py
```

Quy trình này sẽ:
- Trích xuất dữ liệu từ CSV
- Làm sạch và chuyển đổi dữ liệu
- Xây dựng các bảng dimension
- Load dữ liệu vào data warehouse

**Lưu ý**: Quá trình này có thể mất vài phút tùy vào hiệu năng máy tính.

### Bước 5: Khởi chạy Dashboard

```bash
streamlit run app.py
```

Dashboard sẽ mở trong trình duyệt tại: http://localhost:8501

## 📊 Tính năng Dashboard

### 1. Key Performance Indicators (KPIs)
- Total Revenue (Tổng doanh thu)
- Unique Customers (Số khách hàng duy nhất)
- Total Products Sold (Tổng sản phẩm đã bán)
- Total Orders (Tổng số đơn hàng)
- Average Transaction Value (Giá trị giao dịch trung bình)

### 2. Revenue Trend Analysis
- Biểu đồ xu hướng doanh thu theo tháng
- Phân tích số lượng đơn hàng theo thời gian

### 3. Geographic Analysis
- Bản đồ nhiệt toàn cầu (Choropleth Map)
- Top 10 quốc gia theo doanh thu
- Phân tích theo khu vực

### 4. Product Performance
- Top 10 sản phẩm bán chạy theo doanh thu
- Top 10 sản phẩm theo số lượng
- Biểu đồ cột ngang với màu sắc phân biệt

### 5. Customer Analysis
- Phân bổ khách hàng theo quốc gia
- Revenue per customer analysis
- Biểu đồ bánh (Pie chart)

### 6. Time-Based Analysis
- Revenue theo ngày trong tuần
- Weekend vs Weekday analysis
- Orders by day of week

### 7. Data Warehouse Summary
- Bảng tổng hợp số lượng records trong từng bảng
- Metadata về ETL timestamps

## 🔧 Cấu hình

Tất cả cấu hình được quản lý trong file `config.py`:

```python
# Database
DATABASE_NAME = "retail_dwh.db"

# File paths
RAW_DATA_FILE = "online_retail.csv"

# Data quality thresholds
MIN_QUANTITY = 1
MAX_QUANTITY = 100000
MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 50000.00

# ETL settings
BATCH_SIZE = 10000
```

## 📝 Quy trình ETL

### Extract Phase
- Đọc dữ liệu từ CSV file
- Đảm bảo encoding đúng (ISO-8859-1)

### Transform Phase
1. **Data Type Conversion**
   - Chuyển InvoiceDate sang datetime
   - Convert CustomerID sang string
   - Xử lý numeric columns

2. **Data Quality Checks**
   - Lọc bỏ cancelled orders (InvoiceNo bắt đầu bằng 'C')
   - Loại bỏ quantity <= 0 hoặc > threshold
   - Loại bỏ UnitPrice <= 0 hoặc > threshold
   - Xử lý missing data (CustomerID null)

3. **Data Enrichment**
   - Tính TotalRevenue = Quantity * UnitPrice
   - Trích xuất date components (Year, Month, Quarter, etc.)
   - Tạo surrogate keys cho dimensions

4. **Dimension Building**
   - DimDate: Trích xuất từ InvoiceDate
   - DimLocation: Unique countries
   - DimCustomer: Unique customers với country
   - DimProduct: Unique products

5. **Fact Table Preparation**
   - Merge để lấy surrogate keys
   - Validate referential integrity
   - Chuẩn bị fact records

### Load Phase
- Load dimension tables trước (đảm bảo referential integrity)
- Load fact table sau
- Batch insert để tối ưu performance

## 🎓 Kiến thức áp dụng

Dự án này áp dụng các khái niệm và best practices sau:

1. **Data Warehousing Concepts**
   - Star Schema
   - Surrogate Keys
   - Dimension and Fact Tables
   - Referential Integrity

2. **Data Quality**
   - Data validation
   - Outlier detection
   - Missing data handling
   - Business rule enforcement

3. **ETL Best Practices**
   - Logging và error handling
   - Batch processing
   - Idempotency
   - Data lineage tracking

4. **Visualization**
   - Interactive dashboards
   - Multiple chart types
   - Filtering và drilling
   - Responsive design

## 🛠️ Công nghệ sử dụng

- **Python 3.8+**: Ngôn ngữ lập trình chính
- **SQLAlchemy**: ORM cho database operations
- **Pandas**: Data manipulation và analysis
- **Streamlit**: Web framework cho dashboard
- **Plotly**: Interactive visualizations
- **SQLite**: Relational database

## 📈 Kết quả mong đợi

Sau khi chạy ETL, bạn sẽ có:

- ✅ Database với ~40,000-50,000 fact records (sau filtering)
- ✅ 4 dimension tables với đầy đủ attributes
- ✅ Dashboard với 7 sections phân tích
- ✅ Log file ghi lại toàn bộ ETL process

## 🔍 Troubleshooting

### Lỗi: FileNotFoundError for online_retail.csv
**Giải pháp**: Đảm bảo file CSV được đặt đúng path: `data/raw/online_retail.csv`

### Lỗi: Database is locked
**Giải pháp**: Đóng tất cả connections trước khi chạy ETL lại

### Dashboard không hiển thị dữ liệu
**Giải pháp**: 
1. Kiểm tra database đã được tạo chưa
2. Chạy lại ETL pipeline
3. Restart Streamlit app

## 📝 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

## 👤 Author

Built with ❤️ for Data Engineering and Business Intelligence enthusiasts.

---

**Happy Data Engineering! 🚀**
